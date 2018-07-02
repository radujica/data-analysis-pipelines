all: udf_yearmonth

udf_yearmonth:
	clang++-6.0 -shared -fPIC -w udf_yearmonth.cpp -o udf_yearmonth.so

clean:
	rm -rf udf_yearmonth.so
