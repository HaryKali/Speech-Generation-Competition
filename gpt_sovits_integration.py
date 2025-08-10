#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI夏令营音频生成竞赛 - GPT-SoVITS集成方案
基于GPT-SoVITS模型的语音合成替代方案
"""

import os
import sys
import subprocess
import pandas as pd
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


@dataclass
class GPTSoVITSConfig:
    """GPT-SoVITS配置类"""
    model_version: str = "v4"  # v1, v2, v3, v4, v2Pro
    language: str = "zh"  # zh, ja, en, ko, yue
    audio_dir: str = "audio_files"
    result_dir: str = "result"
    max_workers: int = 4
    timeout: int = 300
    log_level: str = "INFO"
    output_format: str = "wav"
    
    # GPT-SoVITS特定配置
    gpt_sovits_path: str = "GPT-SoVITS"  # GPT-SoVITS项目路径
    webui_port: int = 9880
    inference_port: int = 9881


class GPTSoVITSGenerator:
    """GPT-SoVITS音频生成器"""
    
    def __init__(self, config: GPTSoVITSConfig, logger):
        self.config = config
        self.logger = logger
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None
        }
    
    def check_gpt_sovits_installation(self) -> bool:
        """检查GPT-SoVITS安装"""
        try:
            # 检查GPT-SoVITS目录
            if not os.path.exists(self.config.gpt_sovits_path):
                self.logger.error(f"❌ GPT-SoVITS目录不存在: {self.config.gpt_sovits_path}")
                return False
            
            # 检查必要的文件
            required_files = [
                "webui.py",
                "GPT_SoVITS/inference_webui.py",
                "GPT_SoVITS/pretrained_models"
            ]
            
            for file_path in required_files:
                full_path = os.path.join(self.config.gpt_sovits_path, file_path)
                if not os.path.exists(full_path):
                    self.logger.error(f"❌ 缺少必要文件: {full_path}")
                    return False
            
            self.logger.info("✓ GPT-SoVITS安装检查通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GPT-SoVITS检查失败: {e}")
            return False
    
    def start_gpt_sovits_webui(self) -> bool:
        """启动GPT-SoVITS WebUI"""
        try:
            # 切换到GPT-SoVITS目录
            original_dir = os.getcwd()
            os.chdir(self.config.gpt_sovits_path)
            
            # 启动WebUI
            cmd = [
                sys.executable, "webui.py", 
                self.config.language
            ]
            
            self.logger.info(f"🚀 启动GPT-SoVITS WebUI: {' '.join(cmd)}")
            
            # 在后台启动WebUI
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待WebUI启动
            time.sleep(10)
            
            # 检查进程是否还在运行
            if process.poll() is None:
                self.logger.info("✓ GPT-SoVITS WebUI启动成功")
                os.chdir(original_dir)
                return True
            else:
                self.logger.error("❌ GPT-SoVITS WebUI启动失败")
                os.chdir(original_dir)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 启动GPT-SoVITS WebUI失败: {e}")
            os.chdir(original_dir)
            return False
    
    def synthesize_with_gpt_sovits(self, task_row: pd.Series) -> Dict[str, Any]:
        """使用GPT-SoVITS进行语音合成"""
        utt_id = task_row['utt']
        ref_audio = task_row['reference_speech']
        text = task_row['text']
        
        result = {
            "utt_id": utt_id,
            "success": False,
            "error": None,
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            # 构建参考音频路径
            ref_audio_path = os.path.join(self.config.audio_dir, ref_audio)
            if not os.path.exists(ref_audio_path):
                result["error"] = f"参考音频文件不存在: {ref_audio_path}"
                return result
            
            # 使用GPT-SoVITS API进行合成
            # 这里需要根据GPT-SoVITS的API接口进行调整
            cmd = [
                sys.executable, 
                os.path.join(self.config.gpt_sovits_path, "GPT_SoVITS/inference_webui.py"),
                "--ref_audio", ref_audio_path,
                "--text", text,
                "--language", self.config.language,
                "--output", os.path.join(self.config.result_dir, f"{utt_id}.{self.config.output_format}")
            ]
            
            # 执行合成
            subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=self.config.timeout
            )
            
            result["success"] = True
            
        except subprocess.TimeoutExpired:
            result["error"] = f"任务超时 ({self.config.timeout}秒)"
        except subprocess.CalledProcessError as e:
            result["error"] = f"GPT-SoVITS执行失败: {e.stderr}"
        except Exception as e:
            result["error"] = f"未知错误: {str(e)}"
        finally:
            result["duration"] = time.time() - start_time
        
        return result
    
    def process_tasks(self, task_data: pd.DataFrame, max_tasks: Optional[int] = None) -> bool:
        """处理任务"""
        if max_tasks:
            task_data = task_data.head(max_tasks)
            self.logger.warning(f"⚠️  仅处理前{max_tasks}个任务（用于测试）")
        
        self.stats["total"] = len(task_data)
        self.stats["start_time"] = time.time()
        
        self.logger.info(f"开始处理{self.stats['total']}个语音合成任务...")
        
        # 启动GPT-SoVITS WebUI
        if not self.start_gpt_sovits_webui():
            return False
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(self.synthesize_with_gpt_sovits, row): row 
                for _, row in task_data.iterrows()
            }
            
            for future in as_completed(future_to_task):
                task_row = future_to_task[future]
                try:
                    result = future.result()
                    utt_id = result["utt_id"]
                    
                    if result["success"]:
                        self.stats["success"] += 1
                        self.logger.info(f"✓ 任务{utt_id}完成 (耗时: {result['duration']:.2f}s)")
                    else:
                        self.stats["failed"] += 1
                        self.logger.error(f"❌ 任务{utt_id}失败: {result['error']}")
                        
                except Exception as e:
                    self.stats["failed"] += 1
                    self.logger.error(f"❌ 任务处理异常: {e}")
        
        self.stats["end_time"] = time.time()
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPT-SoVITS集成方案")
    parser.add_argument("--version", type=str, default="v4", 
                       choices=["v1", "v2", "v3", "v4", "v2Pro"],
                       help="GPT-SoVITS版本")
    parser.add_argument("--language", type=str, default="zh",
                       choices=["zh", "ja", "en", "ko", "yue"],
                       help="语言")
    parser.add_argument("--max-tasks", type=int, default=10,
                       help="最大处理任务数")
    parser.add_argument("--gpt-sovits-path", type=str, default="GPT-SoVITS",
                       help="GPT-SoVITS项目路径")
    
    args = parser.parse_args()
    
    # 配置
    config = GPTSoVITSConfig(
        model_version=args.version,
        language=args.language,
        gpt_sovits_path=args.gpt_sovits_path
    )
    
    # 日志
    logging.basicConfig(level=getattr(logging, config.log_level))
    logger = logging.getLogger("GPTSoVITS")
    
    logger.info("=" * 60)
    logger.info("GPT-SoVITS集成方案")
    logger.info("=" * 60)
    
    # 初始化生成器
    generator = GPTSoVITSGenerator(config, logger)
    
    # 检查安装
    if not generator.check_gpt_sovits_installation():
        return
    
    # 加载任务数据
    try:
        task_data = pd.read_csv("aigc_speech_generation_tasks.csv")
        logger.info(f"✓ 成功加载任务数据，共{len(task_data)}个任务")
    except Exception as e:
        logger.error(f"❌ 加载任务数据失败: {e}")
        return
    
    # 处理任务
    if not generator.process_tasks(task_data, args.max_tasks):
        return
    
    # 打印统计
    total_time = generator.stats["end_time"] - generator.stats["start_time"]
    print(f"\n🎉 处理完成！")
    print(f"总任务数: {generator.stats['total']}")
    print(f"成功: {generator.stats['success']}")
    print(f"失败: {generator.stats['failed']}")
    print(f"成功率: {generator.stats['success']/generator.stats['total']*100:.1f}%")
    print(f"总耗时: {total_time:.2f}秒")


if __name__ == "__main__":
    main()
