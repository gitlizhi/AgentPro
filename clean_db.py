"""
数据库与文件清理脚本 — 清理工单、编排计划、聊天记录、检查点、临时文件等。

用法：
    # 查看所有表的统计信息（不删除）
    python clean_db.py --stats

    # 清理工单
    python clean_db.py --tickets              # 所有工单（含终态）
    python clean_db.py --tickets --active-only # 仅活跃工单

    # 清理编排计划
    python clean_db.py --orchestration

    # 清理聊天
    python clean_db.py --chat                 # 所有聊天消息
    python clean_db.py --conversations        # 所有会话线程（级联删除消息）

    # 清理短期记忆（LangGraph 检查点）
    python clean_db.py --checkpoints

    # 按智能体清理（聊天 + 检查点 + 会话 + 关联工单 + 关联子任务）
    python clean_db.py --agent agent_main

    # 清理提醒
    python clean_db.py --reminders

    # 清理群聊房间
    python clean_db.py --rooms               # 所有房间
    python clean_db.py --room room_xxx       # 指定房间

    # 清理临时文件
    python clean_db.py --screenshots          # 浏览器截图
    python clean_db.py --tool-outputs         # 工具输出日志
    python clean_db.py --temp                 # 所有临时文件

    # 一键全清（危险，需确认）
    python clean_db.py --all
    python clean_db.py --all --force          # 跳过确认
"""
import asyncio
import argparse
import os
import sys
import time
import shutil
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, os.path.dirname(__file__))


def _get_db():
    """惰性导入数据库模块（避免 --help 时触发 psycopg 依赖错误）。"""
    from agent.db import get_pool, init_db_pool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    return get_pool, init_db_pool, AsyncPostgresSaver

PROJECT_ROOT = Path(__file__).parent

# ── 临时文件目录 ──────────────────────────────────────────────────────────
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
CHAT_IMAGES_DIR = PROJECT_ROOT / "chat_images"
TOOL_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "agent_temp" / "tool_outputs"
SANDBOX_WORKSPACES_DIR = PROJECT_ROOT / "agent" / "agent_temp" / "sandbox_workspaces"
BROWSER_DATA_DIR = PROJECT_ROOT / "browser_data"
REFLECTIONS_DIR = PROJECT_ROOT / "agent" / "data" / "reflections"
AGENT_CONTEXT_DIR = PROJECT_ROOT / "agent" / "agent_context"


# ═══════════════════════════════════════════════════════════════════════════
# 统计
# ═══════════════════════════════════════════════════════════════════════════

async def show_stats():
    """显示所有表的行数和临时文件统计。"""
    get_pool, init_db_pool, _ = _get_db()
    await init_db_pool()
    pool = get_pool()

    tables = [
        "chat_messages",
        "conversation_threads",
        "delegation_tickets",
        "orchestration_plans",
        "orchestration_subtasks",
        "reminders",
        "rooms",
        "room_members",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    ]

    print("\n" + "=" * 70)
    print("  📊 数据库统计")
    print("=" * 70)
    total_rows = 0
    async with pool.connection() as conn:
        for table in tables:
            try:
                rows = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = (await rows.fetchone())[0]
                if count > 0:
                    tag = "📋" if "chat" in table else "🎫" if "ticket" in table else "📦" if "orchestration" in table else "⏰" if "reminder" in table else "🏠" if "room" in table else "🧠" if "checkpoint" in table else "  "
                    print(f"  {tag} {table:<30} {count:>8,} 行")
                    total_rows += count
                else:
                    print(f"    {table:<30}      0 行")
            except Exception:
                print(f"    {table:<30}    (表不存在)")

        # 活跃工单统计
        try:
            active = await conn.execute(
                "SELECT COUNT(*) FROM delegation_tickets "
                "WHERE state NOT IN ('closed','declined','timed_out','cancelled')"
            )
            active_count = (await active.fetchone())[0]
            if active_count:
                print(f"\n  🔴 活跃工单: {active_count} 个")
        except Exception:
            pass

        # 活跃编排计划
        try:
            plans = await conn.execute(
                "SELECT COUNT(*) FROM orchestration_plans "
                "WHERE state NOT IN ('completed','partially_completed','failed','cancelled')"
            )
            plan_count = (await plans.fetchone())[0]
            if plan_count:
                print(f"  🔴 活跃编排计划: {plan_count} 个")
        except Exception:
            pass

    # 文件统计
    print("\n" + "-" * 50)
    print("  📁 文件统计")
    print("-" * 50)
    file_dirs = {
        "screenshots": SCREENSHOTS_DIR,
        "chat_images": CHAT_IMAGES_DIR,
        "tool_outputs": TOOL_OUTPUTS_DIR,
        "sandbox_workspaces": SANDBOX_WORKSPACES_DIR,
        "browser_data": BROWSER_DATA_DIR,
        "reflections": REFLECTIONS_DIR,
        "agent_context": AGENT_CONTEXT_DIR,
    }
    for name, d in file_dirs.items():
        if d.exists():
            files = list(d.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            if file_count > 0:
                print(f"  📄 {name:<20} {file_count:>6,} 文件  ({size_mb:.1f} MB)")
            else:
                print(f"    {name:<20}      0 文件")
        else:
            print(f"    {name:<20}    (目录不存在)")

    print("-" * 50)
    print(f"  数据库总行数: {total_rows:,}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 工单清理
# ═══════════════════════════════════════════════════════════════════════════

async def clean_tickets(active_only: bool = False):
    """清理委托工单，并级联清理关联的编排子任务（避免重启时恢复僵尸计划）。"""
    get_pool, init_db_pool, _ = _get_db()
    await init_db_pool()
    pool = get_pool()
    async with pool.connection() as conn:
        if active_only:
            # 清理引用已删除工单的子任务
            await conn.execute(
                "DELETE FROM orchestration_subtasks WHERE ticket_id IN "
                "(SELECT ticket_id FROM delegation_tickets "
                "WHERE state NOT IN ('closed','declined','timed_out','cancelled'))"
            )
            result = await conn.execute(
                "DELETE FROM delegation_tickets "
                "WHERE state NOT IN ('closed','declined','timed_out','cancelled')"
            )
        else:
            await conn.execute("DELETE FROM orchestration_subtasks")
            result = await conn.execute("DELETE FROM delegation_tickets")
        ticket_count = result.rowcount
        tag = "活跃" if active_only else "所有"

        # 清理没有子任务的空编排计划
        empty_result = await conn.execute(
            "DELETE FROM orchestration_plans "
            "WHERE plan_id NOT IN (SELECT DISTINCT plan_id FROM orchestration_subtasks)"
        )
        empty_count = empty_result.rowcount

        print(f"✅ 已清理 {tag}工单: {ticket_count} 条")
        if empty_count > 0:
            print(f"   级联清理空编排计划: {empty_count} 个")


# ═══════════════════════════════════════════════════════════════════════════
# 编排计划清理
# ═══════════════════════════════════════════════════════════════════════════

async def clean_orchestration(active_only: bool = False):
    """清理编排计划和子任务，并级联清理关联的委托工单。"""
    get_pool, init_db_pool, _ = _get_db()
    await init_db_pool()
    pool = get_pool()
    async with pool.connection() as conn:
        if active_only:
            # 先清理关联工单
            await conn.execute(
                "DELETE FROM delegation_tickets WHERE ticket_id IN "
                "(SELECT ticket_id FROM orchestration_subtasks WHERE plan_id IN "
                "(SELECT plan_id FROM orchestration_plans "
                "WHERE state NOT IN ('completed','partially_completed','failed','cancelled')))"
            )
            await conn.execute(
                "DELETE FROM orchestration_subtasks WHERE plan_id IN "
                "(SELECT plan_id FROM orchestration_plans "
                "WHERE state NOT IN ('completed','partially_completed','failed','cancelled'))"
            )
            result = await conn.execute(
                "DELETE FROM orchestration_plans "
                "WHERE state NOT IN ('completed','partially_completed','failed','cancelled')"
            )
        else:
            await conn.execute("DELETE FROM delegation_tickets")
            await conn.execute("DELETE FROM orchestration_subtasks")
            result = await conn.execute("DELETE FROM orchestration_plans")
        plan_count = result.rowcount
        tag = "活跃" if active_only else "所有"
        print(f"✅ 已清理 {tag}编排计划: {plan_count} 个（子任务和关联工单已级联删除）")


# ═══════════════════════════════════════════════════════════════════════════
# 聊天清理
# ═══════════════════════════════════════════════════════════════════════════

async def clean_chat():
    """清理所有聊天消息（保留会话线程元数据）。"""
    get_pool, init_db_pool, _ = _get_db()
    await init_db_pool()
    pool = get_pool()
    async with pool.connection() as conn:
        result = await conn.execute("DELETE FROM chat_messages")
        count = result.rowcount
        print(f"✅ 已清理聊天消息: {count} 条")


async def clean_conversations():
    """清理所有会话线程（级联删除聊天消息）。"""
    get_pool, init_db_pool, _ = _get_db()
    await init_db_pool()
    pool = get_pool()
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM chat_messages")
        result = await conn.execute("DELETE FROM conversation_threads")
        count = result.rowcount
        print(f"✅ 已清理会话线程: {count} 个（消息已级联删除）")


# ═══════════════════════════════════════════════════════════════════════════
# 检查点清理
# ═══════════════════════════════════════════════════════════════════════════

async def clean_checkpoints(thread_id: str = None):
    """清理 LangGraph 短期记忆检查点。"""
    get_pool, init_db_pool, AsyncPostgresSaver = _get_db()
    await init_db_pool()
    pool = get_pool()
    checkpointer = AsyncPostgresSaver(pool)

    if thread_id:
        await checkpointer.adelete_thread(thread_id)
        print(f"✅ 已清理线程检查点: {thread_id}")
        return

    async with pool.connection() as conn:
        rows = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [row[0] for row in (await rows.fetchall() or [])]

    for tid in thread_ids:
        print(f"  清理检查点: {tid}...")
        await checkpointer.adelete_thread(tid)

    print(f"✅ 已清理检查点: {len(thread_ids)} 个线程")


# ═══════════════════════════════════════════════════════════════════════════
# 按智能体清理
# ═══════════════════════════════════════════════════════════════════════════

async def clean_agent(agent_id: str):
    """清理指定智能体的所有数据：聊天、检查点、会话、关联工单和子任务。"""
    get_pool, init_db_pool, AsyncPostgresSaver = _get_db()
    await init_db_pool()
    pool = get_pool()
    pattern = f"private_{agent_id}_%"
    checkpointer = AsyncPostgresSaver(pool)

    # 1. 聊天消息
    async with pool.connection() as conn:
        result = await conn.execute(
            "DELETE FROM chat_messages WHERE thread_id LIKE %s", (pattern,)
        )
        msg_count = result.rowcount

    # 2. 会话线程
    async with pool.connection() as conn:
        result = await conn.execute(
            "DELETE FROM conversation_threads WHERE agent_id = %s", (agent_id,)
        )
        conv_count = result.rowcount

    # 3. 检查点
    async with pool.connection() as conn:
        rows = await conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE %s", (pattern,)
        )
        thread_ids = [row[0] for row in (await rows.fetchall() or [])]
    for tid in thread_ids:
        await checkpointer.adelete_thread(tid)

    # 4. 关联工单（issuer 或 assignee 是该智能体）
    async with pool.connection() as conn:
        result = await conn.execute(
            "DELETE FROM delegation_tickets WHERE issuer = %s OR assignee = %s",
            (agent_id, agent_id),
        )
        ticket_count = result.rowcount

    # 5. 关联的子任务
    async with pool.connection() as conn:
        result = await conn.execute(
            "DELETE FROM orchestration_subtasks WHERE assigned_to = %s", (agent_id,)
        )
        st_count = result.rowcount
        # 清理没有子任务的空计划
        await conn.execute(
            "DELETE FROM orchestration_plans WHERE issuer = %s "
            "AND plan_id NOT IN (SELECT DISTINCT plan_id FROM orchestration_subtasks)",
            (agent_id,),
        )

    # 6. 提醒
    async with pool.connection() as conn:
        result = await conn.execute(
            "DELETE FROM reminders WHERE user_id = %s", (agent_id,)
        )
        rem_count = result.rowcount

    print(f"✅ 已清理智能体 '{agent_id}':")
    print(f"   聊天消息: {msg_count} 条")
    print(f"   会话线程: {conv_count} 个")
    print(f"   检查点: {len(thread_ids)} 个")
    print(f"   关联工单: {ticket_count} 个")
    print(f"   关联子任务: {st_count} 个")
    print(f"   提醒: {rem_count} 个")


# ═══════════════════════════════════════════════════════════════════════════
# 提醒清理
# ═══════════════════════════════════════════════════════════════════════════

async def clean_reminders():
    get_pool, init_db_pool, _ = _get_db()
    await init_db_pool()
    pool = get_pool()
    async with pool.connection() as conn:
        result = await conn.execute("DELETE FROM reminders")
        count = result.rowcount
        print(f"✅ 已清理提醒: {count} 条")


# ═══════════════════════════════════════════════════════════════════════════
# 房间清理
# ═══════════════════════════════════════════════════════════════════════════

async def clean_rooms(room_id: str = None):
    get_pool, init_db_pool, AsyncPostgresSaver = _get_db()
    await init_db_pool()
    pool = get_pool()
    checkpointer = AsyncPostgresSaver(pool)

    if room_id:
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM room_members WHERE room_id = %s", (room_id,))
            await conn.execute("DELETE FROM rooms WHERE room_id = %s", (room_id,))
        await checkpointer.adelete_thread(f"group_{room_id}")
        print(f"✅ 已清理房间: {room_id}")
        return

    async with pool.connection() as conn:
        rows = await conn.execute("SELECT room_id FROM rooms")
        room_ids = [row[0] for row in (await rows.fetchall() or [])]
        await conn.execute("DELETE FROM room_members")
        result = await conn.execute("DELETE FROM rooms")
        count = result.rowcount
    for rid in room_ids:
        await checkpointer.adelete_thread(f"group_{rid}")
    print(f"✅ 已清理所有房间: {count} 个")


# ═══════════════════════════════════════════════════════════════════════════
# 文件清理
# ═══════════════════════════════════════════════════════════════════════════

def _rm_dir(dir_path: Path, name: str) -> int:
    """删除目录下所有文件，保留目录本身。返回删除的文件数。"""
    if not dir_path.exists():
        print(f"   {name}: 目录不存在，跳过")
        return 0
    deleted = 0
    for item in dir_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
            deleted += 1
        except OSError as e:
            print(f"   ⚠️ 无法删除 {item}: {e}")
    print(f"✅ 已清理 {name}: {deleted} 个文件/目录")
    return deleted


def clean_screenshots():
    _rm_dir(SCREENSHOTS_DIR, "screenshots")


def clean_tool_outputs():
    _rm_dir(TOOL_OUTPUTS_DIR, "tool_outputs")


def clean_temp_files():
    """清理所有临时文件。"""
    print("\n清理临时文件...")
    clean_screenshots()
    clean_tool_outputs()
    # 也清理 agent_temp 下的其他文件
    temp_root = PROJECT_ROOT / "agent" / "agent_temp"
    for item in temp_root.iterdir() if temp_root.exists() else []:
        if item.is_dir() and item.name != "tool_outputs":
            shutil.rmtree(item, ignore_errors=True)
            print(f"✅ 已清理临时目录: {item.name}")
    # 清理项目根目录下的 __pycache__
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    print("✅ 临时文件清理完成")


# ═══════════════════════════════════════════════════════════════════════════
# 一键全清
# ═══════════════════════════════════════════════════════════════════════════

async def clean_all(force: bool = False):
    """清理所有数据：数据库全部表 + 临时文件。"""
    if not force:
        print("\n" + "!" * 70)
        print("  ⚠️  警告：这将删除所有数据库记录和临时文件！")
        print("  包括：聊天记录、工单、编排计划、检查点、房间、提醒、")
        print("        会话线程、截图、工具输出等")
        print("!" * 70)
        confirm = input("  确认输入 'yes' 继续: ")
        if confirm.lower() != "yes":
            print("  已取消。")
            return

    get_pool, init_db_pool, AsyncPostgresSaver = _get_db()
    await init_db_pool()
    pool = get_pool()

    tables = [
        "chat_messages",
        "conversation_threads",
        "delegation_tickets",
        "orchestration_subtasks",
        "orchestration_plans",
        "reminders",
        "room_members",
        "rooms",
    ]

    async with pool.connection() as conn:
        for table in tables:
            try:
                result = await conn.execute(f"DELETE FROM {table}")
                count = result.rowcount
                print(f"  ✅ {table}: {count} 行已删除")
            except Exception as e:
                print(f"  ⚠️ {table}: {e}")

    # 清理检查点（需要专用 API）
    checkpointer = AsyncPostgresSaver(pool)
    async with pool.connection() as conn:
        rows = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [row[0] for row in (await rows.fetchall() or [])]
    for tid in thread_ids:
        await checkpointer.adelete_thread(tid)
    print(f"  ✅ checkpoints: {len(thread_ids)} 个线程已清理")

    # 临时文件
    clean_temp_files()

    print("\n✅ 全部清理完成！")


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="数据库与文件清理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python clean_db.py --stats                    查看所有统计
  python clean_db.py --tickets                  清理所有工单
  python clean_db.py --tickets --active-only    仅清理活跃工单
  python clean_db.py --orchestration            清理所有编排计划
  python clean_db.py --agent agent_main         清理指定智能体的全部数据
  python clean_db.py --chat --checkpoints       清理聊天和检查点
  python clean_db.py --all --force              一键全清（跳过确认）
        """,
    )

    # ── 查看 ──
    parser.add_argument("--stats", action="store_true", help="显示数据库表和临时文件统计")

    # ── 数据库清理 ──
    parser.add_argument("--tickets", action="store_true", help="清理委托工单")
    parser.add_argument("--orchestration", action="store_true", help="清理编排计划和子任务")
    parser.add_argument("--chat", action="store_true", help="清理聊天消息")
    parser.add_argument("--conversations", action="store_true", help="清理会话线程（级联删除消息）")
    parser.add_argument("--checkpoints", action="store_true", help="清理 LangGraph 短期记忆")
    parser.add_argument("--reminders", action="store_true", help="清理提醒")
    parser.add_argument("--rooms", action="store_true", help="清理所有群聊房间")
    parser.add_argument("--agent", type=str, help="清理指定智能体的全部数据")

    # ── 范围限定 ──
    parser.add_argument("--room", type=str, help="指定房间 ID")
    parser.add_argument("--thread", type=str, help="指定检查点线程 ID")
    parser.add_argument("--active-only", action="store_true", help="仅清理活跃（非终态）工单/计划")

    # ── 文件清理 ──
    parser.add_argument("--screenshots", action="store_true", help="清理浏览器截图")
    parser.add_argument("--tool-outputs", action="store_true", help="清理工具输出日志")
    parser.add_argument("--temp", action="store_true", help="清理所有临时文件")

    # ── 全清 ──
    parser.add_argument("--all", action="store_true", help="一键清理所有数据")
    parser.add_argument("--force", action="store_true", help="跳过确认提示")

    args = parser.parse_args()

    # ── 统计 ──
    if args.stats:
        asyncio.run(show_stats())
        sys.exit(0)

    # ── 全清 ──
    if args.all:
        asyncio.run(clean_all(force=args.force))
        sys.exit(0)

    # ── 逐项清理 ──
    any_action = False

    if args.agent:
        asyncio.run(clean_agent(args.agent))
        any_action = True

    if args.tickets:
        asyncio.run(clean_tickets(active_only=args.active_only))
        any_action = True

    if args.orchestration:
        asyncio.run(clean_orchestration(active_only=args.active_only))
        any_action = True

    if args.chat:
        asyncio.run(clean_chat())
        any_action = True

    if args.conversations:
        asyncio.run(clean_conversations())
        any_action = True

    if args.checkpoints:
        if args.thread:
            asyncio.run(clean_checkpoints(thread_id=args.thread))
        else:
            asyncio.run(clean_checkpoints())
        any_action = True

    if args.reminders:
        asyncio.run(clean_reminders())
        any_action = True

    if args.rooms:
        asyncio.run(clean_rooms(room_id=args.room))
        any_action = True

    if args.screenshots:
        clean_screenshots()
        any_action = True

    if args.tool_outputs:
        clean_tool_outputs()
        any_action = True

    if args.temp:
        clean_temp_files()
        any_action = True

    if not any_action:
        parser.print_help()
        print("\n💡 提示：使用 --stats 查看当前数据概况，再决定清理哪些。")
