#!/usr/bin/env python3
"""
Jupytext 批量同步脚本
将指定文件夹下的所有 .ipynb 文件与 .py 文件配对并同步
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_jupytext_command(ipynb_path: Path, formats: str = "ipynb,py:percent", 
                         sync_only: bool = False) -> tuple:
    """
    执行 jupytext 命令
    
    Args:
        ipynb_path: .ipynb 文件路径
        formats: 配对格式，默认为 "ipynb,py:percent"
        sync_only: 如果为 True，只执行同步；否则先设置格式再同步
    
    Returns:
        (success: bool, message: str)
    """
    try:
        # 构建命令
        if sync_only:
            # 仅同步（假设已经配对过）
            cmd = ["jupytext", "--sync", str(ipynb_path)]
        else:
            # 先设置配对格式，然后同步
            cmd = ["jupytext", "--set-formats", formats, "--sync", str(ipynb_path)]
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return (True, f"✅ 成功: {ipynb_path.name}")
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        return (False, f"❌ 失败: {ipynb_path.name} - {error_msg}")
    except Exception as e:
        return (False, f"❌ 异常: {ipynb_path.name} - {str(e)}")


def find_ipynb_files(directory: Path, recursive: bool = True) -> list:
    """
    查找目录下的所有 .ipynb 文件
    
    Args:
        directory: 目标目录
        recursive: 是否递归查找子目录
    
    Returns:
        .ipynb 文件路径列表
    """
    pattern = "**/*.ipynb" if recursive else "*.ipynb"
    return list(directory.glob(pattern))


def process_single_file(args: tuple) -> tuple:
    """
    处理单个文件的包装函数（用于多进程）
    
    Args:
        args: (ipynb_path, formats, sync_only, force)
    """
    ipynb_path, formats, sync_only, force = args
    
    # 检查是否已存在配对的 .py 文件
    py_path = ipynb_path.with_suffix('.py')
    
    if sync_only and not py_path.exists() and not force:
        # 如果要求仅同步但 .py 不存在，跳过
        return (False, f"⏭️  跳过: {ipynb_path.name} (未配对，使用 --force 强制处理)")
    
    return run_jupytext_command(ipynb_path, formats, sync_only)


def main():
    parser = argparse.ArgumentParser(
        description="批量将 .ipynb 文件与 .py 文件配对并同步",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法：处理当前目录下所有 .ipynb 文件
  python jupytext_sync.py
  
  # 指定目录
  python jupytext_sync.py ./notebooks
  
  # 递归处理（默认）
  python jupytext_sync.py ./projects --recursive
  
  # 仅同步已配对的文件（不创建新配对）
  python jupytext_sync.py ./notebooks --sync-only
  
  # 强制处理所有文件（包括未配对的）
  python jupytext_sync.py ./notebooks --sync-only --force
  
  # 使用不同格式
  python jupytext_sync.py ./notebooks --formats "ipynb,py:light"
  
  # 并行处理（4个进程）
  python jupytext_sync.py ./notebooks --workers 4
        """
    )
    
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="目标文件夹路径（默认为当前目录）"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=True,
        help="递归处理子目录（默认启用）"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归处理子目录"
    )
    parser.add_argument(
        "-f", "--formats",
        default="ipynb,py:percent",
        help='配对格式（默认: "ipynb,py:percent"）。可选: py:light, py:percent, py:hydrogen, md, Rmd 等'
    )
    parser.add_argument(
        "-s", "--sync-only",
        action="store_true",
        help="仅同步已配对的文件，不创建新配对"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制处理（与 --sync-only 配合使用时，对未配对文件也执行设置格式）"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="并行处理的进程数（默认: 1）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，只显示将要处理的文件而不实际执行"
    )
    
    args = parser.parse_args()
    
    # 处理 --no-recursive 参数
    recursive = not args.no_recursive
    
    # 验证目录
    target_dir = Path(args.directory).resolve()
    if not target_dir.exists():
        print(f"错误: 目录不存在: {target_dir}")
        sys.exit(1)
    if not target_dir.is_dir():
        print(f"错误: 不是有效目录: {target_dir}")
        sys.exit(1)
    
    # 查找文件
    print(f"🔍 正在查找 {target_dir} 下的 .ipynb 文件...")
    ipynb_files = find_ipynb_files(target_dir, recursive)
    
    if not ipynb_files:
        print("⚠️  未找到任何 .ipynb 文件")
        sys.exit(0)
    
    print(f"📚 找到 {len(ipynb_files)} 个 .ipynb 文件")
    
    # 试运行模式
    if args.dry_run:
        print("\n📋 试运行模式 - 将要处理的文件:")
        for f in ipynb_files:
            py_file = f.with_suffix('.py')
            status = "已配对" if py_file.exists() else "未配对"
            print(f"  • {f.relative_to(target_dir)} [{status}]")
        sys.exit(0)
    
    # 准备任务参数
    task_args = [
        (f, args.formats, args.sync_only, args.force) 
        for f in ipynb_files
    ]
    
    # 处理文件
    print(f"\n🚀 开始处理 (格式: {args.formats}, 进程数: {args.workers})...")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    if args.workers > 1:
        # 并行处理
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_file, arg): arg for arg in task_args}
            
            for future in as_completed(futures):
                success, message = future.result()
                print(message)
                
                if success:
                    success_count += 1
                elif "跳过" in message:
                    skip_count += 1
                else:
                    fail_count += 1
    else:
        # 串行处理
        for task_arg in task_args:
            success, message = process_single_file(task_arg)
            print(message)
            
            if success:
                success_count += 1
            elif "跳过" in message:
                skip_count += 1
            else:
                fail_count += 1
    
    # 输出统计
    print("-" * 60)
    print(f"\n📊 处理完成:")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {fail_count}")
    print(f"   ⏭️  跳过: {skip_count}")
    print(f"   📁 总计: {len(ipynb_files)}")
    
    # 如果有失败，返回非零退出码
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()