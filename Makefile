ligandizer/_vsepr.so: ligandizer/_vsepr.c
	gcc -Wall -O2 -fPIC -shared -o ligandizer/_vsepr.so ligandizer/_vsepr.c
