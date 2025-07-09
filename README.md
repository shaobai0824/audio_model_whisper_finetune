"# audio_model_whisper_finetune" 
使用train.py 前 先登入huggingface # 確保您已在終端機使用 `huggingface-cli login` 登入
終端機下指令可以用tensorboard看儀表板

tensorboard --logdir=./whisper-small-zh-finetune-final

電腦顯示卡的顯存比較高的可以試著將
per_device_train_batch_size=4,
per_device_eval_batch_size=4
同步調高

dataloader_num_workers=0,這個一開始可以嘗試用4開始但是如果發生程式碼沒有跑出
epoch steps value 之類的output 的話 將這個值調低
甚至到0
否則程式會死當不能執行
通常會顯示batch值的增加
程式內部是預設為5000個batch
