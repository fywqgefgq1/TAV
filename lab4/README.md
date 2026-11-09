запустить без тестов: python3 funny_ir_opt.py путь\к\файлу.funny --opts a,b,c | all | none
с тестами: pytest test_funny_ir_opt.py --opts a,b,c | all | none
a,b,c - оптимизации из [const, alg, copy, cse, dce, cfg]
