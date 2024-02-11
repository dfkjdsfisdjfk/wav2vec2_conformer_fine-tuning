from transformers import Wav2Vec2Processor, AutoFeatureExtractor, AutoTokenizer, AutoModelForCTC
from datasets import load_dataset
import torch
from utils.DataCollatorCTCWithPadding import DataCollatorCTCWithPadding
import IPython.display as ipd
import nlptutti as metrics
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
import os
import gc

# 평가에 사용할 성능 지표 계산 함수
def compute_metrics(pred):
    wer_metric = 0
    cer_metric = 0
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    
    for i in range(len(pred_str)):
        wer = metrics.get_wer(pred_str[i], label_str[i])['wer']
        cer = metrics.get_cer(pred_str[i], label_str[i])['cer']
        wer_metric += wer
        cer_metric += cer
        
    wer_metric = wer_metric/len(pred_str)
    cer_metric = cer_metric/len(pred_str)
    
    return {"wer": wer_metric, "cer": cer_metric}


if __name__ =="__main__":
    
    # gpu 캐시 초기화
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["WANDB_DISABLED"] = "true" # wandb 사용하지 않음 / tensorboard 사용

    # 특징 추출기 및 토크나이저 불러오기
    feature_extractor = AutoFeatureExtractor.from_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/feature_extractor_conformer')
    tokenizer = AutoTokenizer.from_pretrained('C:/Users/jjw28/OneDrive/바탕 화면/wav2vec2/model/feature_extractor_tokenizer')

    # 프로세서 정의
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 파인튜닝할 허깅페이스 모델 불러오기
    model = AutoModelForCTC.from_pretrained('42MARU/ko-spelling-wav2vec2-conformer-del-1s',
                                            ctc_loss_reduction="mean", 
                                            pad_token_id=processor.tokenizer.pad_token_id,
                                            )

    # 학습을 위한 데이터컬렉터 함수 정의
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 학습에 상요할 데이터 불러오기
    data = load_dataset('./data/pre_datasets')

    # 모델에 특징 추출 부분 파라미터 업로드 하지 않음
    model.freeze_feature_encoder()

    # 허깅페이스 trainer에 사용할 파라미터들 설정
    training_args = TrainingArguments(
        output_dir='./model/trained_model',
        group_by_length=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        num_train_epochs=30,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500, 
        save_total_limit=1,
        push_to_hub=False,
        dataloader_pin_memory=False,
        logging_dir='./logs',
        load_best_model_at_end=True,
        report_to = "tensorboard",
        )

    # 허깅페이스 trainer 설정 
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        tokenizer=processor.feature_extractor,
        )

    # 학습
    trainer.train()