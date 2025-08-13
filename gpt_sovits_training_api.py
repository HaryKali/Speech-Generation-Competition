import time
from gpt_sovits_training_api import GPTSoVITSTrainingAPI


def main():
    """主函数"""
    print("=" * 60)
    print("GPT-SoVITS API训练脚本")
    print("基于API方式调用GPT-SoVITS进行模型训练")
    print("=" * 60)
    
    # 初始化训练器
    trainer = GPTSoVITSTrainingAPI(
        webui_port=9880,
        version="v2Pro"
    )
    
    # 1. 设置目录
    print("\n1. 设置目录...")
    trainer.setup_directories()
    
    # 2. 加载训练数据
    print("\n2. 加载训练数据...")
    task_data = trainer.load_training_data()
    if task_data is None:
        print("❌ 无法加载训练数据，程序退出")
        return
    
    # 3. 准备训练文件
    print("\n3. 准备训练文件...")
    # 注意：这里设置为只处理前10个文件作为示例
    # 如果要处理所有200个文件，请将max_files设为None
    if not trainer.prepare_training_files(task_data, max_files=10):
        print("❌ 准备训练文件失败，程序退出")
        return
    
    # 获取实验名称
    exp_name = f"training_{int(time.time())}"
    exp_dir = f"{trainer.exp_root}/{exp_name}"
    
    # 4. 启动WebUI（如果需要）
    print("\n4. 检查WebUI状态...")
    if not trainer.start_webui():
        print("⚠️  WebUI启动失败，但继续尝试训练流程")
    
    # 5. 运行数据准备流程
    print("\n5. 运行数据准备流程...")
    inp_text = f"{exp_dir}/2-name2text.txt"
    inp_wav_dir = "./audio_files"
    
    if not trainer.run_data_preparation(exp_name, inp_text, inp_wav_dir):
        print("❌ 数据准备流程失败，程序退出")
        return
    
    # 6. 启动GPT训练
    print("\n6. 启动GPT训练...")
    if not trainer.start_gpt_training(exp_name, batch_size=4, epochs=5):
        print("❌ GPT训练失败，程序退出")
        return
    
    # 7. 启动SoVITS训练
    print("\n7. 启动SoVITS训练...")
    if not trainer.start_sovits_training(exp_name, batch_size=4, epochs=5):
        print("❌ SoVITS训练失败，程序退出")
        return
    
    # 8. 生成训练报告
    print("\n8. 生成训练报告...")
    if not trainer.generate_training_report(exp_name, task_data):
        print("⚠️  生成训练报告失败")
    
    # 9. 创建结果压缩包
    print("\n9. 创建结果压缩包...")
    if not trainer.create_result_archive(exp_name):
        print("⚠️  创建压缩包失败")
    
    print("\n" + "=" * 60)
    print("🎉 GPT-SoVITS训练完成！")
    print(f"📁 实验目录: {exp_dir}")
    print(f"📊 训练报告: result/training_report_{exp_name}.json")
    print(f"📦 结果压缩包: result/result_{exp_name}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
