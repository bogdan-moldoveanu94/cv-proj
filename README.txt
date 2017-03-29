Bogdan Andrei Moldoveanu
This is the first assignment for the Computer Vision course at UNIVE, spring 2017.

This project uses C++11 features(I have updated the cmake file in order to tell g++ this).
I have also added a resource folder with the image provided in the assignment so the make run command would work.
The project should build using the below steps, as in the original template provided here:
https://gitlab.com/fibe/lab_ocv_template.git


mkdir build
cd build
cmake ../ -DOpenCV_DIR="<insert the path of your opencv/build directory>"
make


How to run:

cd build
make run


or, alternatively:

make install
cd dist/bin
invoke the generated executable

