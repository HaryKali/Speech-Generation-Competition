#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI夏令营音频生成竞赛 - 增强版启动脚本
支持F5TTS和GPT-SoVITS两种模型选择
好版本正在更新中
"""


import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """打印项目横幅"""
    print("=" * 60)
    print("🎤 AI夏令营音频生成竞赛")
    print("🚀 增强版 Baseline 启动器")
    print("=" * 60)


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    
    # 检查必要文件
    required_files = [
        "aigc_speech_generation_tasks.csv",
        "audio_files"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            return False
    
    # 检查音频文件数量
    audio_count = len(list(Path("audio_files").glob("*.wav")))
    if audio_count != 200:
        print(f"⚠️  音频文件数量异常: {audio_count}/200")
    
    print("✅ 环境检查通过")
    return True


def check_model_availability():
    """检查模型可用性"""
    models = {}
    
    # 检查F5TTS
    try:
        result = subprocess.run(
            ["f5-tts_infer-cli", "--help"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        models["F5TTS"] = result.returncode == 0
    except:
        models["F5TTS"] = False
    
    # 检查GPT-SoVITS
    models["GPT-SoVITS"] = os.path.exists("GPT-SoVITS")
    
    return models


def show_model_menu():
    """显示模型选择菜单"""
    models = check_model_availability()
    
    print("\n🤖 请选择语音合成模型:")
    print("=" * 40)
    
    if models["F5TTS"]:
        print("1. 🎯 F5TTS (原始baseline)")
        print("   - 优点: 稳定可靠，资源要求低")
        print("   - 缺点: 音质一般，功能有限")
    else:
        print("1. ❌ F5TTS (未安装)")
    
    if models["GPT-SoVITS"]:
        print("2. 🚀 GPT-SoVITS (推荐)")
        print("   - 优点: 音质优秀，功能丰富，多语言支持")
        print("   - 缺点: 资源要求较高，需要预训练模型")
    else:
        print("2. ❌ GPT-SoVITS (未安装)")
        print("   💡 提示: 运行 'git clone https://github.com/RVC-Boss/GPT-SoVITS.git' 安装")
    
    print("3. 📊 查看项目信息")
    print("4. ❌ 退出")
    
    return input("\n请输入选项 (1-4): ").strip()


def show_task_menu():
    """显示任务选择菜单"""
    print("\n📋 请选择运行模式:")
    print("1. 🧪 测试模式 (处理10个任务)")
    print("2. 🚀 完整模式 (处理所有200个任务)")
    print("3. ⚙️  自定义模式")
    print("4. 🔙 返回模型选择")
    
    return input("\n请输入选项 (1-4): ").strip()


def run_f5tts_test_mode():
    """运行F5TTS测试模式"""
    print("\n🧪 启动F5TTS测试模式...")
    cmd = [sys.executable, "baseline_optimized.py", "--max-tasks", "10"]
    subprocess.run(cmd)


def run_f5tts_full_mode():
    """运行F5TTS完整模式"""
    print("\n🚀 启动F5TTS完整模式...")
    print("⚠️  这将处理所有200个任务，可能需要较长时间")
    confirm = input("确认继续? (y/N): ").strip().lower()
    
    if confirm == 'y':
        cmd = [sys.executable, "baseline_optimized.py", "--all"]
        subprocess.run(cmd)
    else:
        print("❌ 已取消")


def run_f5tts_custom_mode():
    """运行F5TTS自定义模式"""
    print("\n⚙️  F5TTS自定义模式配置:")
    
    # 获取任务数量
    while True:
        try:
            max_tasks = input("处理任务数量 (1-200): ").strip()
            if max_tasks.lower() == 'all':
                max_tasks = None
                break
            max_tasks = int(max_tasks)
            if 1 <= max_tasks <= 200:
                break
            print("❌ 请输入1-200之间的数字")
        except ValueError:
            print("❌ 请输入有效数字")
    
    # 获取并发数量
    while True:
        try:
            workers = input("并发线程数 (1-16, 默认4): ").strip()
            if not workers:
                workers = 4
                break
            workers = int(workers)
            if 1 <= workers <= 16:
                break
            print("❌ 请输入1-16之间的数字")
        except ValueError:
            print("❌ 请输入有效数字")
    
    # 构建命令
    cmd = [sys.executable, "baseline_optimized.py", "--workers", str(workers)]
    if max_tasks:
        cmd.extend(["--max-tasks", str(max_tasks)])
    else:
        cmd.append("--all")
    
    print(f"\n🚀 启动命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_gpt_sovits_test_mode():
    """运行GPT-SoVITS测试模式"""
    print("\n🧪 启动GPT-SoVITS测试模式...")
    
    # 选择版本
    print("\n选择GPT-SoVITS版本:")
    print("1. V4 (推荐，最高音质)")
    print("2. V2Pro (平衡性能和成本)")
    print("3. V2 (基础版本)")
    
    version_choice = input("请选择版本 (1-3): ").strip()
    version_map = {"1": "v4", "2": "v2Pro", "3": "v2"}
    version = version_map.get(version_choice, "v4")
    
    cmd = [sys.executable, "gpt_sovits_integration.py", "--version", version, "--max-tasks", "10"]
    subprocess.run(cmd)


def run_gpt_sovits_full_mode():
    """运行GPT-SoVITS完整模式"""
    print("\n🚀 启动GPT-SoVITS完整模式...")
    print("⚠️  这将处理所有200个任务，可能需要较长时间")
    confirm = input("确认继续? (y/N): ").strip().lower()
    
    if confirm == 'y':
        # 选择版本
        print("\n选择GPT-SoVITS版本:")
        print("1. V4 (推荐，最高音质)")
        print("2. V2Pro (平衡性能和成本)")
        print("3. V2 (基础版本)")
        
        version_choice = input("请选择版本 (1-3): ").strip()
        version_map = {"1": "v4", "2": "v2Pro", "3": "v2"}
        version = version_map.get(version_choice, "v4")
        
        cmd = [sys.executable, "gpt_sovits_integration.py", "--version", version, "--max-tasks", "200"]
        subprocess.run(cmd)
    else:
        print("❌ 已取消")


def run_gpt_sovits_custom_mode():
    """运行GPT-SoVITS自定义模式"""
    print("\n⚙️  GPT-SoVITS自定义模式配置:")
    
    # 选择版本
    print("\n选择GPT-SoVITS版本:")
    print("1. V4 (推荐，最高音质)")
    print("2. V2Pro (平衡性能和成本)")
    print("3. V2 (基础版本)")
    
    version_choice = input("请选择版本 (1-3): ").strip()
    version_map = {"1": "v4", "2": "v2Pro", "3": "v2"}
    version = version_map.get(version_choice, "v4")
    
    # 获取任务数量
    while True:
        try:
            max_tasks = input("处理任务数量 (1-200): ").strip()
            if max_tasks.lower() == 'all':
                max_tasks = 200
                break
            max_tasks = int(max_tasks)
            if 1 <= max_tasks <= 200:
                break
            print("❌ 请输入1-200之间的数字")
        except ValueError:
            print("❌ 请输入有效数字")
    
    # 选择语言
    print("\n选择语言:")
    print("1. 中文 (zh)")
    print("2. 英语 (en)")
    print("3. 日语 (ja)")
    print("4. 韩语 (ko)")
    print("5. 粤语 (yue)")
    
    lang_choice = input("请选择语言 (1-5): ").strip()
    lang_map = {"1": "zh", "2": "en", "3": "ja", "4": "ko", "5": "yue"}
    language = lang_map.get(lang_choice, "zh")
    
    cmd = [
        sys.executable, "gpt_sovits_integration.py", 
        "--version", version, 
        "--language", language,
        "--max-tasks", str(max_tasks)
    ]
    
    print(f"\n🚀 启动命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def show_project_info():
    """显示项目信息"""
    print("\n📊 项目信息:")
    print("-" * 40)
    
    # 统计音频文件
    audio_files = list(Path("audio_files").glob("*.wav"))
    print(f"📁 参考音频文件: {len(audio_files)}个")
    
    # 统计任务数据
    if os.path.exists("aigc_speech_generation_tasks.csv"):
        import pandas as pd
        try:
            df = pd.read_csv("aigc_speech_generation_tasks.csv")
            print(f"📋 任务数据: {len(df)}个任务")
            
            # 显示前几个任务示例
            print("\n📝 任务示例:")
            for i, row in df.head(3).iterrows():
                print(f"  任务{row['utt']}: {row['text'][:30]}...")
        except Exception as e:
            print(f"❌ 读取任务数据失败: {e}")
    
    # 检查结果目录
    if os.path.exists("result"):
        result_files = list(Path("result").glob("*.wav"))
        print(f"🎵 已生成音频: {len(result_files)}个")
    
    # 检查模型可用性
    models = check_model_availability()
    print(f"\n🤖 模型状态:")
    print(f"  F5TTS: {'✅ 可用' if models['F5TTS'] else '❌ 未安装'}")
    print(f"  GPT-SoVITS: {'✅ 可用' if models['GPT-SoVITS'] else '❌ 未安装'}")
    
    print("-" * 40)


def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请检查项目文件")
        return
    
    # 主循环
    while True:
        choice = show_model_menu()
        
        if choice == '1':
            # F5TTS选项
            if not check_model_availability()["F5TTS"]:
                print("❌ F5TTS未安装，请先安装F5TTS")
                input("按回车键继续...")
                continue
            
            while True:
                task_choice = show_task_menu()
                
                if task_choice == '1':
                    run_f5tts_test_mode()
                elif task_choice == '2':
                    run_f5tts_full_mode()
                elif task_choice == '3':
                    run_f5tts_custom_mode()
                elif task_choice == '4':
                    break
                else:
                    print("❌ 无效选项，请重新选择")
                
                input("\n按回车键继续...")
        
        elif choice == '2':
            # GPT-SoVITS选项
            if not check_model_availability()["GPT-SoVITS"]:
                print("❌ GPT-SoVITS未安装")
                print("💡 安装命令: git clone https://github.com/RVC-Boss/GPT-SoVITS.git")
                input("按回车键继续...")
                continue
            
            while True:
                task_choice = show_task_menu()
                
                if task_choice == '1':
                    run_gpt_sovits_test_mode()
                elif task_choice == '2':
                    run_gpt_sovits_full_mode()
                elif task_choice == '3':
                    run_gpt_sovits_custom_mode()
                elif task_choice == '4':
                    break
                else:
                    print("❌ 无效选项，请重新选择")
                
                input("\n按回车键继续...")
        
        elif choice == '3':
            show_project_info()
            input("\n按回车键继续...")
        
        elif choice == '4':
            print("\n👋 再见！")
            break
        
        else:
            print("❌ 无效选项，请重新选择")
            input("按回车键继续...")


if __name__ == "__main__":
    main()
