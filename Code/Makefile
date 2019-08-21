CC = g++
CFLAGS = -g -Wall
SRCS = Image_stabilization.cpp
PROG = Image_stabilization

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)
TIME = -lrt

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) $(TIME)

clean:
	rm -f *.o Image_stabilization