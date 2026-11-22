void f(int);
void (*volatile hook)(int);
void run(void) { hook = f; hook(0); }
