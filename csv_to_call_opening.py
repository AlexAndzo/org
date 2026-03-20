"""
步骤1：将 CSV 中 cold_call_opening_line 写入数据库 call_opening
- 匹配规则：CSV 的 Company Name 与数据库 contact_company 完全相同
- 数据库配置在 call_match_package/config.py 的 DB_CONFIG
- CSV 路径在 call_match_package/config.py 的 CSV_FILE
"""

import pandas as pd
import pymysql
from loguru import logger
from pathlib import Path

import config

# Configure file logging - logs to call_match_package/logs directory
LOGS_DIR = Path(__file__).parent / "logs"

# Try to set up file logging, but don't crash if we don't have permission
try:
    LOGS_DIR.mkdir(exist_ok=True)
    logger.add(
        LOGS_DIR / "step1_csv_to_opening_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        encoding="utf-8",
    )
    logger.info(f"[STEP1] File logging enabled: {LOGS_DIR}")
except PermissionError as e:
    logger.warning(f"[STEP1] Cannot write to log directory {LOGS_DIR}: {e}. Logs will only appear in console.")
except Exception as e:
    logger.warning(f"[STEP1] Failed to set up file logging: {e}. Logs will only appear in console.")


def get_conn():
    cfg = config.DB_CONFIG.copy()
    cfg["cursorclass"] = pymysql.cursors.DictCursor
    return pymysql.connect(**cfg)


def load_csv(path: str) -> pd.DataFrame:
    logger.info(f"[STEP1] 正在加载 CSV 文件: {path}")
    encodings = ["utf-8", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"[STEP1] CSV 加载成功 | 编码: {enc} | 总行数: {len(df)}")
            return df
        except Exception as e:
            logger.debug(f"[STEP1] 尝试编码 {enc} 失败: {e}")
            last_err = e
            continue
    logger.error(f"[STEP1] CSV 加载失败，所有编码都不可用: {last_err}")
    raise last_err


def main():
    logger.info("=" * 60)
    logger.info("[STEP1] === CSV -> call_opening 更新开始 ===")
    logger.info("=" * 60)

    df = load_csv(config.CSV_FILE)

    required_cols = {"Company Name", "cold_call_opening_line"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"[STEP1] CSV 缺少必要列: {missing}")
        raise ValueError(f"CSV 缺少列: {missing}")

    logger.info(f"[STEP1] CSV 列检查通过 | 必要列: {required_cols}")

    df_valid = df[df["cold_call_opening_line"].notna()].copy()
    df_invalid = df[df["cold_call_opening_line"].isna()]

    logger.info(f"[STEP1] 有效行数（有 opening line）: {len(df_valid)}")
    logger.info(f"[STEP1] 无效行数（无 opening line）: {len(df_invalid)}")

    conn = get_conn()
    logger.info("[STEP1] 数据库连接成功")

    updated_total = 0
    rows_processed = 0
    skipped_empty_company = 0
    skipped_empty_opening = 0
    no_match_companies = []
    success_companies = []

    try:
        with conn.cursor() as cur:
            total_rows = len(df_valid)

            for idx, row in df_valid.iterrows():
                rows_processed += 1
                company = row["Company Name"]
                opening = row["cold_call_opening_line"]

                # 检查空值
                if pd.isna(company) or str(company).strip() == "":
                    skipped_empty_company += 1
                    logger.debug(
                        f"[STEP1] 行 {rows_processed}/{total_rows} | ⏭ 跳过 | "
                        f"原因: Company Name 为空"
                    )
                    continue

                if pd.isna(opening) or str(opening).strip() == "":
                    skipped_empty_opening += 1
                    logger.debug(
                        f"[STEP1] 行 {rows_processed}/{total_rows} | ⏭ 跳过 | "
                        f"Company={company} | 原因: opening line 为空"
                    )
                    continue

                company = str(company).strip()
                opening = str(opening).strip()

                # 执行更新
                cur.execute(
                    """
                    UPDATE sdr_cdr_inout_data
                    SET call_opening = %s
                    WHERE contact_company = %s
                    """,
                    (opening, company),
                )
                rows_affected = cur.rowcount

                if rows_affected > 0:
                    updated_total += rows_affected
                    success_companies.append(company)
                    logger.info(
                        f"[STEP1] 行 {rows_processed}/{total_rows} | ✓ 写入成功 | "
                        f"Company={company} | 更新行数: {rows_affected} | "
                        f"Opening: {opening[:50]}{'...' if len(opening) > 50 else ''}"
                    )
                else:
                    no_match_companies.append(company)
                    logger.warning(
                        f"[STEP1] 行 {rows_processed}/{total_rows} | ⚠ 无匹配记录 | "
                        f"Company={company} | 数据库中找不到该公司"
                    )

                # 每100条提交一次
                if rows_processed % 100 == 0:
                    conn.commit()
                    logger.debug(f"[STEP1] 已处理 {rows_processed}/{total_rows} 行，中间提交完成")

        conn.commit()
        logger.info("[STEP1] 最终提交完成")

    except Exception as e:
        logger.error(f"[STEP1] 数据库操作失败: {str(e)}")
        raise
    finally:
        conn.close()
        logger.info("[STEP1] 数据库连接已关闭")

    # 汇总报告
    logger.info("=" * 60)
    logger.info("[STEP1] === 更新完成 - 汇总报告 ===")
    logger.info("=" * 60)
    logger.info(f"[STEP1] 📊 CSV 总行数: {len(df)}")
    logger.info(f"[STEP1] 📊 有效行数（有 opening）: {len(df_valid)}")
    logger.info(f"[STEP1] 📊 处理行数: {rows_processed}")
    logger.info(f"[STEP1] ✓ 成功更新公司数: {len(success_companies)}")
    logger.info(f"[STEP1] ✓ 成功更新数据库行数: {updated_total}")
    logger.info(f"[STEP1] ⏭ 跳过 - Company 为空: {skipped_empty_company}")
    logger.info(f"[STEP1] ⏭ 跳过 - Opening 为空: {skipped_empty_opening}")
    logger.info(f"[STEP1] ⚠ 无匹配记录的公司数: {len(no_match_companies)}")

    if no_match_companies:
        logger.info("[STEP1] 无匹配记录的公司列表:")
        for company in no_match_companies[:20]:  # 最多显示20个
            logger.info(f"[STEP1]   - {company}")
        if len(no_match_companies) > 20:
            logger.info(f"[STEP1]   ... 还有 {len(no_match_companies) - 20} 个公司未显示")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
