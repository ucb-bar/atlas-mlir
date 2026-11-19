#include "atlas_host.h"
#include <string.h>
int main(void) {
 unsigned char storage[32]={0}, source[32], actual[32];
 for(unsigned i=0;i<32;i++) source[i]=(unsigned char)(3*i+1);
 if(atlas_host_direct_memory_write(0,(uintptr_t)storage,source,32)) return 1;
 if(atlas_host_direct_memory_read(0,(uintptr_t)storage,actual,32)) return 2;
 if(memcmp(source,actual,32)) return 3;
 if(!atlas_host_direct_memory_read(0,UINTPTR_MAX,actual,2)) return 4;
 if(!atlas_host_direct_memory_write(0,0,source,1)) return 5;
 return 0;
}
