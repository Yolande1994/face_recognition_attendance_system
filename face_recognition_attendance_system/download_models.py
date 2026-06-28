import zipfile
import urllib.request
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent

# 官方原版 buffalo_l 模型压缩包地址
OFFICIAL_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# 模型最终存放目录
TARGET_DIR = ROOT / "models" / "models" / "buffalo_l"

# 完整性校验文件列表
REQUIRED_FILES = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx"
]


def download_with_progress(url: str, save_path: Path):
    """带进度显示的文件下载"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100.0, downloaded / total_size * 100)
        mb_done = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  下载进度: {percent:5.1f}%  [{mb_done:.1f}MB / {mb_total:.1f}MB]", end="")

    print("正在下载 buffalo_l 官方模型包...")
    urllib.request.urlretrieve(url, save_path, reporthook=progress_hook)
    print("\n  下载完成")


def check_files_complete() -> bool:
    """校验目标目录下所有模型文件是否齐全"""
    return all((TARGET_DIR / filename).exists() for filename in REQUIRED_FILES)


def main():
    print("=" * 55)
    print("  InsightFace buffalo_l 模型自动部署工具")
    print(f"  目标路径: {TARGET_DIR}")
    print("=" * 55)

    if check_files_complete():
        print("[√] 所有模型文件已存在，无需重复下载")
        print("[完成] 启动命令: python main.py")
        return

    temp_zip = ROOT / "buffalo_l_temp.zip"

    try:
        # 下载官方压缩包
        download_with_progress(OFFICIAL_ZIP_URL, temp_zip)

        # 解压到 buffalo_l 目录
        print("\n[*] 正在解压文件...")
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(TARGET_DIR)
        print("[√] 解压完成")

    except Exception as e:
        print(f"\n[×] 下载失败: {str(e)}")
        print("\n手动下载地址:")
        print(OFFICIAL_ZIP_URL)
        print("解压到: models/models/buffalo_l/ 目录")
    finally:
        # 清理临时压缩包
        if temp_zip.exists():
            temp_zip.unlink()
            print("[*] 已清理临时文件")

    # 最终完整性校验
    if check_files_complete():
        print("\n" + "=" * 55)
        print("[成功] 5个模型文件全部部署完成，与官方版本一致")
        print("启动命令: python main.py")
        print("=" * 55)
    else:
        print("\n[失败] 文件校验不通过，请检查目录内容")


if __name__ == "__main__":
    main()