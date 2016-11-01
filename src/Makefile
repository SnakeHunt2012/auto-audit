C = gcc
CXX = g++

MYHOME=/home/huangjingwen/work/auto-audit

INC = -I . -I $(MYHOME)/qmodule/qcommon-1.0.0/include -I $(MYHOME)/qmodule/transcode-1.0.0/include -I $(MYHOME)/qmodule/segment-2.2.1/include -I ./jsoncpp -I $(MYHOME)/xgboost/include

OS = $(shell uname)
CFLAGS = -Wall 
#CFLAGS = -std=c++11 -Wall 

CXXFLAGS_NOMDBM = $(CFLAGS)
CFLAGS += $(INC)
CXXFLAGS = $(CFLAGS)

RELEASE = 1

ifeq ($(RELEASE), 1)
	CXXFLAGS += -O3
else
	CXXFLAGS += -g
endif


OBJS = common.o global_dict.o sparse_matrix.o /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o

BINDIR = ./

LIBS = -lpthread -fopenmp -lfcgi -lcurl $(MYHOME)/xgboost/lib/libxgboost.a $(MYHOME)/xgboost/lib/librabit.a $(MYHOME)/xgboost/lib/libdmlc.a $(MYHOME)/qmodule/segment-2.2.1/lib/libsegment.a $(MYHOME)/qmodule/transcode-1.0.0/lib/libtranscode.a

.PHONY: all

all: develop server

develop: develop.o $(OBJS)
	$(CXX) $(CXXFLAGS) $< ${OBJS} -o $(BINDIR)/$@ ${LIBS}

server: server.o classifier.o common.o global_dict.o sparse_matrix.o /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o
	$(CXX) $(CXXFLAGS) $< classifier.o common.o global_dict.o sparse_matrix.o /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o -o $(BINDIR)/$@ ${LIBS}

classifier: classifier.o common.o global_dict.o sparse_matrix.o /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o
	$(CXX) $(CXXFLAGS) $< common.o global_dict.o sparse_matrix.o /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o -o $(BINDIR)/$@ ${LIBS}

common: common.o global_dict.o sparse_matrix.o /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o
	$(CXX) $(CXXFLAGS) $< global_dict.o sparse_matrix.o -o $(BINDIR)/$@ ${LIBS}

global_dict: /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o
	$(CXX) $(CXXFLAGS) $< /home/huangjingwen/work/auto-audit/src/jsoncpp/jsoncpp.o -o $(BINDIR)/$@ ${LIBS}

clean:
	rm -f *.o  *~ /cores/core.* core *.core *.a

stop:
	ps -ef | grep 'lighttpd -f fastcgi.conf' | grep -v 'grep' | awk '{print $$2}' | xargs kill

start: server
	rm -rf auto-audit.fcgi && cp server auto-audit.fcgi && lighttpd -f fastcgi.conf

restart: server stop start
