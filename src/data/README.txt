conda environment -> należy przygotowac środowisko zgodnie z plikiem requirements.txt
należy ustawić ścieżke do podstawowego folderu w pliku utils.py


kolejność wywoływania skryptów python3:
1. arxiv_map_dataset.py
2. doaj_map_dataset.py
3. scimago_clean_dataset.py
4. scimago_map_dataset.py
5. connect_all_datasets.py
6. clean_all_datasets.py

w razie problemów z alokacją pamięci przy kolejnych skryptach należy zabić wszystkie procesy (pkill -u user) i ponownie uruchomić skrpyt. Problem jest potencjalnie do rozwiązania zmieniając zapis plików z csv np do pickle lub innych formatów, powinno działać szybciej, do sprawdzenia. 

