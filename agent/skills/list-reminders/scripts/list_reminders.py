# !/usr/bin/env python
import sys
import argparse
import asyncio
import os
import psycopg
from psycopg.rows import dict_row


async def list_reminders(user_id: str) -> str:
    dsn = os.getenv("POSTGRES_URI")
    if not dsn:
        return "错误：未设置数据库连接字符串"
    
    conn = await psycopg.AsyncConnection.connect(dsn)
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT reminder_time, message FROM reminders WHERE user_id = %s AND NOT triggered ORDER BY reminder_time",
            (user_id,)
        )
        rows = await cur.fetchall()
    await conn.close()
    
    if not rows:
        return "您当前没有未到期的提醒。"
    
    result = "您当前的提醒：\n"
    for row in rows:
        dt = row['reminder_time'].strftime('%Y-%m-%d %H:%M:%S')
        result += f"- {dt}：{row['message']}\n"
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True)
    args = parser.parse_args()
    output = await list_reminders(args.user_id)
    print(output)


if __name__ == "__main__":
    asyncio.run(main())