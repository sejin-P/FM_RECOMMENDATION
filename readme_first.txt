모든 코드와 model의 binary file, 리포트는 criteo_conversion_logs/real 디렉토리 안에 있습니다.
리포트는 README.md 가 아닌 README.pdf 입니다. 유의하시 바랍니다.
fm.py에서 FM Model을 선언하고 model.py에 그에 대한 training을 진행합니다. FM Model은 base_model.py의 Model을 상속받습니다.

main.py의 밑의 주석에서 설명하듯이 dataset 각각에 대해 train 시킬 수 있습니다. 하지만
