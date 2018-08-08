# dl-system-test
Homework in PPCA SJTU
Step0:
Merge the folder with Folder "test"
Step1:
g++ -shared con2d.cpp -o con2d.so -fPIC -lpthread -O3
Step2:
python run_test.py your_model
