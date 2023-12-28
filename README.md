# LLM-Lab

## 프로젝트 구조도
LLM_Lab/
│
├── config/                                 # 전체 설정 관련 모듈들을 포함하는 패키지
│   ├── arguments.py                            # 명령줄 인자 파싱
│   ├── config_builder.py                       # 설정 객체 빌드
│   ├── training_environment.py                 # 훈련 환경 설정
│   ├── models/                                 # 모델 관련 설정
│   │   ├── lora_config.py                          # LoRA 설정
│   │   ├── model_config.py                         # 모델 구성 설정
│   │   └── quantization_config.py                  # 양자화 설정
│   └── training/                               # 훈련 설정
│       ├── tokenizer_config.py                     # AutoTokenizer 파라미터 설정
│       ├── training_config_manager.py              # Transformers TrainingArguments 파라미터 설정
│       ├── training_logging_config.py              # 로깅 구성 설정
│       └── training_config.py                      # 훈련 구성 설정
│
├── utils/                                  # 유틸리티 함수 및 클래스
│   ├── directory_utils.py                      # 디렉토리 관련 유틸리티
│   ├── experiment_datetime_utils.py            # 실험 날짜/시간 유틸리티
│   └── os_environment_utils.py                 # OS 환경 관련 유틸리티
│
├── data/                                   # 데이터 로딩 및 처리
│   └── data_loader.py                          
│
├── models/                                 # 모델 로딩 및 처리
│   └── model_loader.py                         
│
├── prompts/                                # 프롬프트 관련
│   └── prompt_loader.py
│
├── run_QnA_task.sh                         # MRC Q&A Task 실행 스크립트
├── run_mlflow.sh                           # MLFlow 실행 스크립트
├── LLM_QnA_experiment_main.py              # 메인 실행 모듈
└── requirements.txt                        # 의존성 목록
