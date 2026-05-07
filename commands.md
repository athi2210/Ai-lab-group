$env:PATH = "C:\rtools43\x86_64-w64-mingw32.static.posix\bin;" + $env:PATH
cd "C:\Users\athit\OneDrive\Dokumente\Th Nürnberg\1.Semester\Ai Accelarator\Ai Accelerator lab exercise\aia-matmul-template"
gcc -O3 matmul.c -o matmul.exe -lm
.\matmul.exe