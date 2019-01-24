/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*  pipe( feedback(pipe(A,B)), C)
 *
 *    -----            ------            ------ 
 *   |  A  |--------> |  B   |--------> |  C   |
 *   |     |          |      |          |      |
 *    -----            ------            ------ 
 *        ^              |           
 *         \             |           
 *          -------------            
 *
 */

/* Author: Massimo Torquati
 *
 */


#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>  // for ff_monode

using namespace ff;

#define NUMTASKS 10

/* --------------------------------------- */
// NOTE: multi-input node!!!!!!
//       That is because node B is multi-output!
class A: public ff_minode_t<long> {    
public:
    long *svc(long *task) {
        static long k=1;
        if (task==NULL) {
            ff_send_out((long*)k);
            return GO_ON;
        }
        printf("A received %ld back from B\n", (long)task);
        if (k++==NUMTASKS) return NULL;
        return (long*)k;
    }
};
/* --------------------------------------- */

class B: public ff_monode_t<long> {
public:
    long *svc(long *task) {
        const long& t = (long)task;
        printf("B got %ld from A\n", t);
        if (t & 0x1) {
            printf("B sending %ld to C\n", t);
            ff_send_out_to(task, 1);  // sending forward 
        }
        printf("B sending %ld back\n", t);
        ff_send_out_to(task, 0);  // sending back
        return GO_ON;
    }
};
/* --------------------------------------- */

class C: public ff_node_t<long> {
public:
    long *svc(long *t) { 
        printf("C got %ld\n", (long)t);
        return GO_ON;
    }
};
/* --------------------------------------- */

int main() {
    A a;  B b;  C c;

    ff_Pipe<long,long> pipe1(a, b);
    if (pipe1.wrap_around()<0) {
        error("creating feedback channel between b and a\n");
        return -1;
    }

    ff_Pipe<> pipe(pipe1, c);

    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    printf("DONE\n");    
    return 0;
}

