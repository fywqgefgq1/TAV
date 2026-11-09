запустить без тестов: python3 funny_ir_opt.py путь\к\файлу.funny --opts a,b,c | all | none --color-scope all | temps
с тестами: pytest test_funny_ir_opt.py --opts a,b,c | all | none --color-scope all | temps
a,b,c - оптимизации из [const, alg, copy, cse, dce, cfg]
all - граф конфликтов строится для всех переменных
temps - только для временных
