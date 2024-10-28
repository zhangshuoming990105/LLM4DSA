# This file contains the prompts for the AI model to generate the x86 assembly code from the input C code.

emnlp_baseline_prompts = {
    "general": """[INST]compile the following C code into x86_64 assembly.
use AT&T syntax in the assembly.
make sure the output assembly is within "```x86" and "```" tags.
[/INST]""",
}

# emnlp_baseline_prompts = {
#     "general": """[INST]compile the following C code into x86 assembly.
# the system architecture is x86_64. please make sure use AT&T syntax in the assembly.
# input code will be inside "```c" and "```" tags.
# make sure the output assembly is within "```x86" and "```" tags.
# [/INST]""",
# }

one_shot_prompts_for_languages = {
    "python": """you are a professional code compiler, you can translate any code language into assembly code.
Now your job is to compile the following python code into x86 assembly. Using AT&T syntax in the x86 assembly.
You should first infer the type information in the input code, using type hints to rewrite the python code first, you can assume int for most integer values, unless explicit float values are used
and then generate full compilable x86_64 assembly, make sure there are no missing 
You don't need to explain the code.
Example:
#Input:
```python
if __name__ == "__main__":
    print("Hello, World!")
    exit(0)
```
#Output:
```x86
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	endbr64
	pushq	%rbp
	movq	%rsp, %rbp
	leaq	.LC0(%rip), %rdi
	call	puts@PLT
	movl	$0, %eax
	popq	%rbp
	ret
.LC0:
	.string	"Hello, World!"

```
""",
    "c++": """you are a professional code compiler, you can translate any code language into assembly code.
Now your job is to compile the following C++ code into x86 assembly. Using AT&T syntax in the x86 assembly.
There are name mangling and implicit std initializer/constructor/destructor issues that you may be difficult of, so your first job is to convert the C++ code into C code like a code translation. Then compile the C code into x86 assembly. 
Example:
#Input:
```cpp
int foo(vector<int> v) {
    for(int i = 0; i < v.size();i++) {
        printf("%d ", v[i]);
    }
}
```
#C code:
```c
int foo(int v[], int size) {
    for(int i = 0; i < size; i++) {
        printf("%d ", v[i]);
    }
}
```
#Output:
```x86
	.text
	.globl	foo
	.type	foo, @function
foo:
.LFB0:
	endbr64
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	$0, -4(%rbp)
	jmp	.L2
.L3:
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -4(%rbp)
.L2:
	movl	-4(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jl	.L3
	nop
	leave
	ret
.LC0:
	.string	"%d "
```""",
    "cuda": """""",
}

compiler_annotation_prompts = {
    "description": """[INST]I want you to act like a compiler that translate C code into x86 assembly. 
However, I don't want you to do it directly because that's memorizing. I want you to do so by strictly follow my guide and examples.
In order to compile the following code into assembly, we need:
1. first analyze the customized structs types and give them correct offset, size and padding, note that each struct follows the largest alignment basic type in its elements.
2. collect all the constants, name their labels with meaningful names, and all variables with their type to form a SymbolTable.
Remember to find all distinct constants appeared in source, do not precomputing them. For example, 1.0f and 1.0 are different constants, 
and 2.0/1.3 should generate two constants, and compute them in the assembly.
But for same constants(type also the same) appeared multiple times, you should only generate one constant label is enough.
3. compile the code using the above SymbolTable. generate AT&T syntax x86_64 assembly.
[/INST]""",
    "example": """[INST]###Example:
#Input:
```c
#include <stdio.h>

typedef struct {
    int company_id;
    char company_name[10];
} Company;

typedef struct {
    int person_age;
    char person_name[10];
    long person_id;
    Company *person_company;
} Person;

double dval = 1.0;
static int arr[5] = {1, 2, 3, 4, 5};

void foo(Person *person1, Person *person2) {
    printf("enter foo\n");
    if(person1->person_age > person2->person_age) {
        double d = -1.0;
        dval += d;
        person1->person_age += 5;
    } else {
        person2->person_age += 5;
    }
    dval += 2.0;
    arr[0] += 1;
    printf("exit foo\n");
}

```
#Step1, Let's first get the struct annotation:
# 1. struct annotation:
```plaintext
typedef struct {
    int company_id; // offset 0, size 4
    char company_name[10]; // offset 4, size 10, pad 2 to 4 byte alignment
} Company;  // total size 16, alignment 4, 16%4=0
typedef struct {
    int person_age; // offset 0, size 4
    char person_name[10]; // offset 4, size 10, pad 2 to 4 byte alignment
    long person_id; // offset 16, size 8
    Company *person_company; // offset 24, size 8
} Person; // total size 32, alignment 8, 32%8=0
```
#Step2, we should based on the struct annotation, find all symbol instances to generate the SymbolTable:
# 2. SymbolTable:
```plaintext
- Constants:
-- literals:
.LC_enter_foo_str: 
    .string "enter foo\n"
.LC_exit_foo_str:
    .string "exit foo\n"
-- float and double values:
double: 1.0, 2.0, -1.0
float: none
## NOTE that 1.0f and 1.0 are different constants, 
## and 2.0/1.3 should generate two constants, and compute them in the assembly.

- Variables:
-- Global variables: 
    double dval
-- Static variables: 
    int arr[10]
-- Local variables:
    double d
-- Function arguments:
person1: Person *, size 8
person2: Person *, size 8

- Warp these values to generate STACK ALLOCATION(local + arguments):
#double d: -8(%rbp), [-8, 0), size 8
#Person *person1: -16(%rbp), [-16, -8), size 8
#Person *person2: -24(%rbp), [-24, -16), size 8
```
#Step3, now we can compile the code using the SymbolTable.
```x86
    .text
# Global variables
    .globl  dval
    .data
    .align 8
    .type   dval, @object
    .size   dval, 8
dval:
    .double 1.0  # double dval = 1.0;

# Static variables
    .align 16
    .type   arr, @object
    .size   arr, 20
arr:
    .long   1  # static int arr[5] = {1, 2, 3, 4, 5};
    .long   2
    .long   3
    .long   4
    .long   5

# Local constants in function
    .section    .rodata
# String literals
.LC_enter_foo_str:
    .string "enter foo\n"  # For printf("enter foo\n");
.LC_exit_foo_str:
    .string "exit foo\n"   # For printf("exit foo\n");
    .align 8
# Numeric constants
.LC_neg_one:
    .double -1.0  # For double d = -1.0;

    .align 8
.LC_two:
    .double 2.0  # For dval += 2.0;

# Function body
    .text
    .globl  foo
    .type   foo, @function
foo:
.LF_foo_entry:
	# Prologue
    endbr64
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $24, %rsp
    movq    %rdi, -16(%rbp)  # Store person1 pointer
    movq    %rsi, -24(%rbp)  # Store person2 pointer
    
    # printf("enter foo\n");
    leaq    .LC_enter_foo_str(%rip), %rdi
    movb	$0, %al
    call    printf@PLT
    
    # if(person1->person_age > person2->person_age)
    movq    -16(%rbp), %rax
    movl    (%rax), %edx  # person1->person_age
    movq    -24(%rbp), %rax
    movl    (%rax), %eax  # person2->person_age
    cmpl    %eax, %edx
    jle     .L_if1_else
    
    # Inside if block
    movsd   .LC_neg_one(%rip), %xmm0
    movsd   %xmm0, -8(%rbp)  # double d = -1.0;
    movsd   dval(%rip), %xmm0
    addsd   -8(%rbp), %xmm0
    movsd   %xmm0, dval(%rip)  # dval += d;
    
    movq    -16(%rbp), %rax
    movl    (%rax), %eax
    leal    5(%rax), %edx
    movq    -16(%rbp), %rax
    movl    %edx, (%rax)  # person1->person_age += 5;
    jmp     .L_if1_end
    
.L_if1_else:
    # Inside else block
    movq    -24(%rbp), %rax
    movl    (%rax), %eax
    leal    5(%rax), %edx
    movq    -24(%rbp), %rax
    movl    %edx, (%rax)  # person2->person_age += 5;

.L_if1_end:
    # After if-else block
    movsd   dval(%rip), %xmm1
    movsd   .LC_two(%rip), %xmm0
    addsd   %xmm1, %xmm0
    movsd   %xmm0, dval(%rip)  # dval += 2.0;
    
    movl    arr(%rip), %eax
    addl    $1, %eax
    movl    %eax, arr(%rip)  # arr[0] += 1;
    
    # printf("exit foo\n");
    leaq    .LC_exit_foo_str(%rip), %rdi
    movb	$0, %al
    call    printf@PLT
    nop
	
	# Epilogue
    leave
    ret

```
[/INST]""",
}

lego_prompts = {
    "description": """[INST]I want you to act like a compiler that translate C code into x86 assembly. 
However, I don't want you to do it directly because that's memorizing. I want you to do so by strictly follow my guide and examples.
In order to compile the following code into assembly, we need:
1. first analyze the customized structs types and give them correct offset, size and padding, note that each struct follows the largest alignment basic type in its elements.
2. collect all the constants, name their labels with meaningful names, and all variables with their type to form a SymbolTable.
Remember to find all distinct constants appeared in source, do not precomputing them. For example, 1.0f and 1.0 are different constants, 
and 2.0/1.3 should generate two constants, and compute them in the assembly.
But for same constants(type also the same) appeared multiple times, you should only generate one constant label is enough.
3. based on the code control structure, split the program into control blocks(like a if() {}, for() {}, while() {}, or a meaningful group of sequential stmts, etc),
we should first split the code from the outmost blocks, if one controlblock is too large, then we should consider to split it into smaller blocks.
When you split the code, you should maintain the SymbolTable and the constants, and the struct annotation.
And most importantly, you should organize the splitted blocks into a list, order them in the order of the code execution.
Then you should also generate labels and comments to indicate the start and end of each splitted block.
4. Finally, you should output the x86 assembly code based on the splitted blocks, and the SymbolTable.

[/INST]""",
    "example": """[INST]###Example:
#Input:
```c
#include <stdio.h>

typedef struct {
    int company_id;
    char company_name[10];
} Company;

typedef struct {
    int person_age;
    char person_name[10];
    long person_id;
    Company *person_company;
} Person;

double dval = 1.0;
static int arr[5] = {1, 2, 3, 4, 5};

void foo(Person *person1, Person *person2) {
    printf("enter foo\n");
    if(person1->person_age > person2->person_age) {
        double d = -1.0;
        dval += d;
        person1->person_age += 5;
    } else {
        person2->person_age += 5;
    }
    dval += 2.0;
    arr[0] += 1;
    printf("exit foo\n");
}

```
#Step1, Let's first get the struct annotation:
# 1. struct annotation:
```plaintext
typedef struct {
    int company_id; // offset 0, size 4
    char company_name[10]; // offset 4, size 10, pad 2 to 4 byte alignment
} Company;  // total size 16, alignment 4, 16%4=0
typedef struct {
    int person_age; // offset 0, size 4
    char person_name[10]; // offset 4, size 10, pad 2 to 4 byte alignment
    long person_id; // offset 16, size 8
    Company *person_company; // offset 24, size 8
} Person; // total size 32, alignment 8, 32%8=0
```
#Step2, we should based on the struct annotation, find all symbol instances to generate the SymbolTable:
# 2. SymbolTable:
```plaintext
- Constants:
-- literals:
.LC_enter_foo_str: 
    .string "enter foo\n"
.LC_exit_foo_str:
    .string "exit foo\n"
-- float and double values:
double: 1.0, 2.0, -1.0
float: none
## NOTE that 1.0f and 1.0 are different constants, 
## and 2.0/1.3 should generate two constants, and compute them in the assembly.

- Variables:
-- Global variables: 
    double dval
-- Static variables: 
    int arr[10]
-- Local variables:
    double d
-- Function arguments:
person1: Person *, size 8
person2: Person *, size 8

- Warp these values to generate STACK ALLOCATION(local + arguments):
#double d: -8(%rbp), [-8, 0), size 8
#Person *person1: -16(%rbp), [-16, -8), size 8
#Person *person2: -24(%rbp), [-24, -16), size 8
```
#Step3, now we should split the code into control blocks, we do this by annotate the source code first:
## Annotated C code for splitting:
```c
// Block 1, global, static, local_constants initialization
#include <stdio.h>

typedef struct {
    int company_id;
    char company_name[10];
} Company;

typedef struct {
    int person_age;
    char person_name[10];
    long person_id;
    Company *person_company;
} Person;

double dval = 1.0;
static int arr[5] = {1, 2, 3, 4, 5};

void foo(Person *person1, Person *person2) {
    // Block 2, function prologue + followup sequential stmts
    printf("enter foo\n");
    // Block 3, if block 1
    if(person1->person_age > person2->person_age) {
        double d = -1.0;
        dval += d;
        person1->person_age += 5;
    } else {
        person2->person_age += 5;
    }
    // Block 4, followup sequential stmts + function epilogue
    dval += 2.0;
    arr[0] += 1;
    printf("exit foo\n");
}
```
## output block numbers:
```plaintext
1,2,3,4
```
## Then we should generate the splitted blocks one by one:


#Step4, now we can compile the code using the SymbolTable.
```x86
## Block 1-begin
    .text
# Global variables
    .globl  dval
    .data
    .align 8
    .type   dval, @object
    .size   dval, 8
dval:
    .double 1.0  # double dval = 1.0;

# Static variables
    .align 16
    .type   arr, @object
    .size   arr, 20
arr:
    .long   1  # static int arr[5] = {1, 2, 3, 4, 5};
    .long   2
    .long   3
    .long   4
    .long   5

# Local constants in function
    .section    .rodata
# String literals
.LC_enter_foo_str:
    .string "enter foo\n"  # For printf("enter foo\n");
.LC_exit_foo_str:
    .string "exit foo\n"   # For printf("exit foo\n");
    .align 8
# Numeric constants
.LC_neg_one:
    .double -1.0  # For double d = -1.0;

    .align 8
.LC_two:
    .double 2.0  # For dval += 2.0;
## Block 1-end

## Block 2_begin
# Function body
    .text
    .globl  foo
    .type   foo, @function
foo:
.LF_foo_entry:
	# Prologue
    endbr64
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $24, %rsp
    movq    %rdi, -16(%rbp)  # Store person1 pointer
    movq    %rsi, -24(%rbp)  # Store person2 pointer
    
    # printf("enter foo\n");
    leaq    .LC_enter_foo_str(%rip), %rdi
    movb	$0, %al
    call    printf@PLT
## Block 2_end
## Block 3_begin
    
    # if(person1->person_age > person2->person_age)
    movq    -16(%rbp), %rax
    movl    (%rax), %edx  # person1->person_age
    movq    -24(%rbp), %rax
    movl    (%rax), %eax  # person2->person_age
    cmpl    %eax, %edx
    jle     .L_if1_else
    
    # Inside if block
    movsd   .LC_neg_one(%rip), %xmm0
    movsd   %xmm0, -8(%rbp)  # double d = -1.0;
    movsd   dval(%rip), %xmm0
    addsd   -8(%rbp), %xmm0
    movsd   %xmm0, dval(%rip)  # dval += d;
    
    movq    -16(%rbp), %rax
    movl    (%rax), %eax
    leal    5(%rax), %edx
    movq    -16(%rbp), %rax
    movl    %edx, (%rax)  # person1->person_age += 5;
    jmp     .L_if1_end
    
.L_if1_else:
    # Inside else block
    movq    -24(%rbp), %rax
    movl    (%rax), %eax
    leal    5(%rax), %edx
    movq    -24(%rbp), %rax
    movl    %edx, (%rax)  # person2->person_age += 5;

.L_if1_end:
    # After if-else block
## Block 3_end
## Block 4_begin
    movsd   dval(%rip), %xmm1
    movsd   .LC_two(%rip), %xmm0
    addsd   %xmm1, %xmm0
    movsd   %xmm0, dval(%rip)  # dval += 2.0;
    
    movl    arr(%rip), %eax
    addl    $1, %eax
    movl    %eax, arr(%rip)  # arr[0] += 1;
    
    # printf("exit foo\n");
    leaq    .LC_exit_foo_str(%rip), %rdi
    movb	$0, %al
    call    printf@PLT
    nop
	
	# Epilogue
    leave
    ret
## Block 4_end
```
[/INST]
""",
}

compiler_short_prompts = {
    "general": """[INST]you are a helpful AI assistant, you will think carefully and follow the instructions to assist the user.[/INST]""",
    "mission": """[INST]you are a professional AI assistant in code, based on the user input C code, 
you are going to help me to generate the corresponding x86 assembly.
You will perform like a compiler with O0 optimization level, the architecture is x86_64.
We can assume there will only be one function body to be compiled.[/INST]""",
    "code_format": """[INST] input code will be inside "```c" and "```"tags, please also make sure the generated x86 assembly be inside "```x86" and "```" tags.[/INST]""",
    "code_example": """[INST]Example:
#Input:
```c
int main() {
    printf("Hello, World!\n");
    return 0;
}
```
#Output:
```x86
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	endbr64
	pushq	%rbp
	movq	%rsp, %rbp
	leaq	.LC0(%rip), %rdi
	call	puts@PLT
	movl	$0, %eax
	popq	%rbp
	ret
.LC0:
	.string	"Hello, World!"

```[/INST]""",
}

compiler_prompts = {
    "general": """[INST]you are a helpful AI assistant, you will think carefully and follow the instructions to assist the user.[/INST]""",
    "mission": """[INST]you are a professional AI assistant in code, based on the following instructions, 
you are going to help me to generate the corresponding x86 assembly of input c code step by step.
We can assume there will only be one function body to be compiled.[/INST]
""",
    "step1": """[INST]#Step1# rename the variables with the same names but across different scopes.[/INST]
""",
    "step1_example": """[INST]#Step1Input:
```c
int x = 0;
int y = 10;
int foo() {
    return x;
}
int main() {
    int x = 10;
    int sum = y;
    for(int x = 0; x < 10; x++) {
        sum += x;
    }
    return x;
}
```
#Step1Output:
```c
int global_x = 0;
int global_y = 10;
int foo() {
    return global_x;
}
int main() {
    int x_0 = 10;
    int sum = global_y;
    for(int x_1 = 0; x_1 < 10; x_1++) {
        sum += x_1;
    }
    return x_0;
}
```[/INST]
""",
    "step2": """[INST]#Step2# generate a variable binding table output(in plaintext) from the Step1Output,
which binds each variable into a memory address in stack for each function.
let's assume the target platform is x86 assembly. For each variable, 
1) we fill the item with: type, size, alignment, start index(l), end index(r), and we use (l, r] to represent the interval of the variable.
where l starts from 0, and keeps decreasing for new item. r == l - size, l == r + size.
new item's l is first set by last r, then we calculate new item's r by l - size.
such intervals will not overlap, but may not satisfy the alignment requirement.
2) then, we fix alignment, if x is not aligned, we shift (x-size, x] by subtract a value to make x to the multiple of alignment.
so each item is:
variable_name: type, size, alignment, (l, r], representation= r(%rbp)
[/INST]
""",
    "step2_example": """[INST]Example for #Step2#:
#Step2Input:
```c
int slice2(int total, int arr1[]) {
    int sum = 0;
    for(int i = 0; i < total; i++) {
        sum += arr1[i];
    }
    double *not_used = NULL;
    float not_used_2 = 0.0f;
    return sum;
}
```
#Step2Output:
```plaintext
## 1) type, size, alignment, (l, r](check if r % alignment==0), representation
total:  int, size 4, align 4, (0, -4], -4(%rbp)
arr1:  	int pointer, size 8, align 8, (-4, -12](12%8!=0), -12(%rbp)
sum:	int, size 4, align 4, (-12, -16], -16(%rbp)
i:  int, size 4, align 4, (-16, -20], -20(%rbp)
not_used:   double pointer, size 8, align 8, (-20, -28](28%8!=0), -28(%rbp)
not_used_2: float, size 4, align 4, (-28, -32], -32(%rbp)
## 2) fix alignment
total:  int, size 4, align 4, (0, -4], -4(%rbp)
arr1:   int pointer, size 8, align 8, (-8, -16], from (-4, -12]-4=(-8, -16], -16(%rbp)
sum:    int, size 4, align 4, (-16, -20], -20(%rbp)
i:  int, size 4, align 4, (-20, -24], -24(%rbp)
not_used:   double pointer, size 8, align 8, (-24, -32], -32(%rbp)
not_used_2: float, size 4, align 4, (-32, -36], -36(%rbp)
```
[/INST]
""",
    "step3": """[INST]#Step3# generate x86 assembly based on #Step1# and #Step2#'s outputs, 
the renamed source code and the variable binding table,
for each variable in the table: we get its memory address in stack from the table's representation,
The output assembly code should follow the AT&T syntax and uses explicit instruction operations,
like `movl` for 4-byte operations and `movq` for 8-byte operations. 
The code should be well-formatted and include annotations corresponding to the source code.[/INST]
""",
    "step3_example": """[INST]#Step1Output:
```c
float foo(int param_i, int param_j, int arr[]) {
    int sum = param_i;
    int temp = param_j;
    for(int i_1 = 0; i_1 < sum; i_1++) {
        for(int j_1 = 0; j_1 < sum; j_1++) {
            temp += arr[i_1] * arr[j_1];
        }
    }
    return temp;
}
```
#Step2Output:
```plaintext
## 1) type, size, alignment
param_i:  int, size 4, align 4, (0, -4], -4(%rbp)
param_j:  float, size 4, align 4, (-4, -8], -8(%rbp)
arr:    int array pointer, size 8, align 8, (-8, -16], -16(%rbp)
sum: int, size 4, align 4, (-16, -20], -20(%rbp)
temp: float, size 4, align 4, (-20, -24], -24(%rbp)
i_1: int, size 4, align 4, (-24, -28], -28(%rbp)
j_1: int, size 4, align 4, (-28, -32], -32(%rbp)
## 2) fix alignment(final)
param_i:  int, size 4, align 4, (0, -4], -4(%rbp)
param_j:  float, size 4, align 4, (-4, -8], -8(%rbp)
arr:    int array pointer, size 8, align 8, (-8, -16], -16(%rbp)
sum: int, size 4, align 4, (-16, -20], -20(%rbp)
temp: float, size 4, align 4, (-20, -24], -24(%rbp)
i_1: int, size 4, align 4, (-24, -28], -28(%rbp)
j_1: int, size 4, align 4, (-28, -32], -32(%rbp)
```
#FinalOutput:
```x86
    .text
    .globl  foo
foo:
    pushq    %rbp
    movq     %rsp, %rbp
    movl     %edi, -4(%rbp)     # param_i
    movss   %xmm0, -8(%rbp)    # param_j
    movq     %rsi, -16(%rbp)    # arr
    movl     -4(%rbp), %edi
    movl     %edi, -20(%rbp)    # sum = param_i
    cvttss2si   -8(%rbp), %edi
    movl     %edi, -24(%rbp)    # temp = param_j
    movl     $0, -28(%rbp)      # i_1 = 0
.L1:
    movl     -28(%rbp), %eax
    cmpl     -20(%rbp), %eax    # i_1 < sum
    jge     .L2
    movl     $0, -32(%rbp)      # j_1 = 0
.L3:
    movl     -32(%rbp), %eax
    cmpl     -20(%rbp), %eax    # j_1 < sum
    jge     .L4
    movq	-16(%rbp), %rax
	movslq	-28(%rbp), %rcx
	movl	(%rax,%rcx,4), %edx # arr[i_1]
	movq	-16(%rbp), %rax
	movslq	-32(%rbp), %rcx
	imull	(%rax,%rcx,4), %edx # arr[i_1] * arr[j_1]
	addl	-24(%rbp), %edx     
	movl	%edx, -24(%rbp)     # temp += arr[i_1] * arr[j_1]
    addl     $1, -32(%rbp)      # j_1++
    jmp     .L3
.L4:
    addl     $1, -28(%rbp)      # i_1++
    jmp     .L1
.L2:
    cvtsi2ssl   -24(%rbp), %xmm0   # return temp
    popq	%rbp
    retq
```
[/INST]
""",
    "recap": """[INST]Now, you will regard the user's input as the #Step1#'s #Input, and execute #Step1#,#Step2#,#Step3# step by step.
And finally get the x86 assembly we want as the output. Please perform the above steps now. Think carefully when doing each step.
The input will be marked with #Input[/INST]
""",
}

decompiler_prompts = {
    "general": """[INST]you are a helpful AI assistant, you will think carefully and follow the instructions to assist the user.[/INST]
""",
    "mission": """[INST]you are going to help me to decompile the input assembly code into the corresponding C code.[/INST]""",
    "task_format": """[INST]#Description:##A description of task, from assembly language to C language##
#Input:
## Input source assembly code##
#Output:
##Output target C code##[/INST]
""",
    "task_example": """[INST]#Description:decompile the following x86 assembly code into C code.
#Input:
```x86
	.globl	fib
	.type	fib,@function
fib:
# %bb.0:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	movl	%edi, -8(%rbp)
	cmpl	$1, -8(%rbp)
	jg	.LBB0_2
# %bb.1:
	movl	-8(%rbp), %eax
	movl	%eax, -4(%rbp)
	jmp	.LBB0_3
.LBB0_2:
	movl	-8(%rbp), %edi
	subl	$1, %edi
	callq	fib
	movl	%eax, -12(%rbp)                 # 4-byte Spill
	movl	-8(%rbp), %edi
	subl	$2, %edi
	callq	fib
	movl	%eax, %ecx
	movl	-12(%rbp), %eax                 # 4-byte Reload
	addl	%ecx, %eax
	movl	%eax, -4(%rbp)
.LBB0_3:
	movl	-4(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	retq
.Lfunc_end0:
	.size	fib, .Lfunc_end0-fib
```
#Output:
```c
int fib(int n) {
    if(n <= 1) {
        return n;
    }
    int a = fib(n - 1);
    int b = fib(n - 2);
    return a + b;
}
```[/INST]
""",
}

code_translator_prompts = {
    "general": """[INST]you are a helpful AI assistant, you will think carefully and follow the instructions to assist the user.[/INST]
""",
    "mission": """[INST]you are a professional AI assistant in code, based on the following instructions,
you are going to help to translate the input code into another language.[/INST]
""",
    "task_format": """[INST]#Description:##A description of task, from source language to target language##
#Input:
##Input source language code##
#Output:
##Output target language code##[/INST]
""",
    "task_example": """[INST]#Description:translate the following C code into Python code.
#Input:
```c
int sum_list(int arr[], int n) {
    int sum = 0;
    for(int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
```
#Output:
```python
def sum_list(arr: List[int], n: int) -> int:
    sum = 0
    for i in range(n):
        sum += arr[i]
    return sum
```[/INST]
""",
}

fix_prompts = {
    "analyze_pre": """Please analyze the following C code to figure out 
if the following features are included in the code:
"numerical": If the code contains numerical values, like 1.0, 2e-5, 3.14f, etc, if the code only use integers, then don't include this feature.
"hex_octal": If the code contains hex or octal values, like 0x3f, 077, etc.
"recursive": If the code is recursive.
"irregular": If the code has some needle-in-the-haystack, that differs from the normal code.
"long": If the code is pretty long and complex.(more than 50 lines)
"cmp_ins": If the code uses comparison instructions, like >, <, >=, <=, ==, !=.
"div_ins": If the code uses division instructions(/)
"mod_ins": If the code uses modulo instructions(%)
"str": If the code manipulates strings or char arrays.
"order": If the code contains complicated expression(need many operations to evaluate), 
be aware of the order of operations.
""",
    "analyze_post": """Return your answer with whether the code contains the above features.
If the code has some features, list them with the feature name and
separate them with ",". 
Before return, check the **Extra information**, analyze it, and give all possible relevant features to this message.
If you are not sure about the features, just return the certain features.
If the code doesn't have any of the features,
just return ```plaintext\n```.
example output: 
```plaintext
numerical, cmp_ins, div_ins, irregular
```""",
    "error_message_pre": """Previous output is not correct, please check the error message below and correct the code:""",
    "error_message_post": """Based on the **error message**(from stdout and stderr), analyze which part of the code is wrong and fix it.
There are three types, if its a **syntax error**, you should focus on the syntax error and fix it, it's usually your misused of syntax.
if its a **runtime error**, not syntax and no output, you should focus on the logic and find possible segmentation fault, null pointer dereference, etc.
if its a **result error**, where the expected output mismatch with the actual output, you should focus on the logic and find the wrong part of the assembly code.
And if the results are close but not the same, you should focus on the numerical values and the order of operations.
Remember, the source code is always correct, the error is always on the generated assembly code. You should revise the code logic carefully, pay attention to the details.
When you try to fix the assembly, always recap the source code to make sure you are following the source code logic. That's where you can find the error.
NEVER add logic that is not in the source code, always follow the source code logic.
Like the order of operations, the type usage of instructions(do they have implicit conversion), and the irregular part of code(like the needle-in-the-haystack).
always remember the **error message**, it's quite important! Then generate your fixed code with proper comment on the code.
FORMAT: make sure the generated x86 assembly in the "#Output" be inside "```x86" and "```" tags.""",
    "irregular": """For the irregular code, you need to think carefully on the code logic, line by line,
to prevent the code from being miscompiled, for example, for a code snippet like:
```c
if(a == 0 || a == 1 || a== 2 || a== 3 || ... || a == 14 || 15 || a == 16) {
...
```the irregular part is || 15 ||, you should always follow what the code logic is, and generate the assembly accordingly.
here, '||' a none-zero value is always true, you should strictly follow the code logic here to or True, do not compare with a.
""",
    "numerical": """For numerical values(double and float, not integers), you don't need to convert the value to IEEE754 format, 
just keep them as they are, float as .float, double as .double, no need to convert it to int value.
Besides, using meaningful labels for the numbers is also helpful. Also, if the value is like 23.0/1.5, etc, you should generate two values instead of precomputed value.
Also, 23.0f is a float value, 23.0 is a double value, you should generate them with the correct type.
And for numbers, it should be based on with or without f to determine the type.
like, 23.0f is a float, 23.0 is a double, 23 is an int. If assign a double to a float variable, you should load first, then convert, store.
or, when calculate a(a float) + 32.0(a double), you should convert a to double first, add, then convert back to float.
similar for other type conversions as well.
Example:
```c
#include <stdio.h>

float foo(float fahrenheit) {
 return ( 5.0 / 9.0) * (fahrenheit - 32);
}
```
```x86
	.text
	.globl	foo
	.type	foo, @function
foo:
.LFB0:
	endbr64
	pushq	%rbp
	movq	%rsp, %rbp
	movss	%xmm0, -12(%rbp)  # store the input fahrenheit value
	movss	-12(%rbp), %xmm0
	subss	.LC_32(%rip), %xmm0  # subtract 32, promote int 32 to float because it's used to compute with a float value
	movsd	.LC_five(%rip), %xmm1  # load the value 5.0 as a double
	divsd	.LC_nine(%rip), %xmm1  # divide the result by 9.0
	cvtss2sd	%xmm0, %xmm0  # convert the result to double
	mulsd	%xmm1, %xmm0  # multiply the result with 5.0/9.0
	cvtsd2ss	%xmm0, %xmm0  # convert the result back to float
	popq	%rbp
	ret
.LC_32:
	.float 32.0  # 32 as a float, because it's used to compute directly with a float value
.LC_five:
	.double 5.0
.LC_nine:
	.double 9.0
```
Repeat, do not precompute value make meaningful labels, and use the correct type.
""",
    "hex_octal": """For hexadecimal or octal values, you don't need to convert them to base-10 value,
just keep them as they are,
Example:
C: 
    int a = 0x23;
x86:
    movl $0x23, xxx(a's address)
""",
    "cmp_ins": """cmp instructions in x86 assembly are used to compare two values, 
and set the flags register accordingly.
WARNING: we cannot use two memory addresses to be compared, or two values to be compared
Example:'cmpl -4(%rbp), -8(%rbp)' is not allowed, 'cmpl $10, $20' is not allowed.
so you need to load one memory/value to register before comparing.
""",
    "mod_ins": """mod in x86 assembly are used to calculate the remainder of a division operation.
many compilers will optimize such operation to other instructions with arithemetic operations.
however, you are not good at arithemetic computation, so you need to use the mod instruction directly.
WARNING: idivl $10(any constant value) is not allowed, you need to load the value to register first.
for example: 'int a = 10 % 3;' -compile to-> 'movl $10, %eax\n movl $3, %ecx\n idivl %ecx\n movl %edx, xxx(a's address)'
""",
    "div_ins": """div in x86 assembly are used to calculate the quotient of a division operation.
many compilers will optimize such operation to other instructions with arithemetic operations.
however, you are not good at arithemetic computation, so you need to use the div instruction directly.
WARNING: idivl $10(any constant value) is not allowed, you need to load the value to register first.
for example: 'int a = 10 / 3;' ---> 'movl $10, %eax\n movl $3, %ecx\n idivl %ecx\n movl %eax, xxx(a's address)'
""",
    "recursive": """this is a recursive function, you need to generate the assembly with strict stack management.
make sure you have the correct stack frame and stack pointer management.
""",
    "order": """this is a function that contains complex arithmetic operations, be aware of the order of operations.
You need to generate the assembly with strict arithmetic operation order.
*/% has higher priority than +-.
+= has higher priority than <<, >>
<<, >> has higher priority than >, <, >=, <=.
>, <, >=, <= has higher priority than ==, !=.
==, != has higher priority than &&.
&& has higher priority than ||.
Note that () can change the priority of operations. so when generate assembly, think carefully about the order of operations.
You can generate assembly together with comments to show the order of operations.
Use comments on the assembly to help you understand the computation order.
""",
    "str": """for string in C, char *, its length is the string length + 1, because c string ends with \0,
keep in mind with that to avoid string operation mismatch.""", # TBD
    "long": """""", # TBD
}

subtask_prompts = {
    "struct_annotate": """below is a c program that use plenty customized structs, in order to translate the program, i need you to annotate the code. First, find out all structs in this program, annotate every element's offset and size using comments.
example:
typedef struct MAT_PARAMS_S {
  int N;
  short *A;
  short *B;
  int *C;
} mat_params;
-->
typedef struct MAT_PARAMS_S {
  int N; // offset 0, size 4, pad 4
  short *A; // offset 8, size 8,
  short *B; // offset 16, size 8
  int *C; // offset 24, size 8
} mat_params; // total size 32, alignment 8, 32%8=0
alignment requirement:
pointer/double/long need to be 8 byte aligned, int/float need 4, short need 2, char need 1, 
and >8 struct need 8 byte aligned.
So when calculate the final padding of struct(with >8 size), make sure it is pad to multiple of 8.
C program is below, think carefully and annotate, make sure you return the full code, do not omit any part.
You only need to annotate the struct definition part, no need to annotate the function part.
""",
    "stack_allocation":"""Your ultimate goal is to translate the c program into x86_64 assembly. Now from your annotation, you currently need to assign the stack allocation for these variables appeared in the program based on the annotated structs. make sure every variable uses the proper size and offset.
Now you don't need to compile it, just return the stack you assigned for local variables.
example:
```c
void foo() {
    char c = 1;
    int b = 2;
    double d[4] = {1.0, 2.0, 0.0, 2.0};
}
```
-->
allocation:
```plaintext
c: -1(%rbp), [0, -1), size 1, 
b: -8(%rbp), [-8, -16), size 8, pad 7 to 8 byte alignment
d: -48(%rbp), [-16, -48), size 32
```""",
    "complicated_example":"""
```c
#include <stdint.h>
#include <stdio.h>
typedef unsigned short u16;
typedef unsigned char u8;
typedef struct list_data_s {
  short data16;
  short idx;
} list_data;

typedef struct list_head_s {
  struct list_head_s *next;
  struct list_data_s *info;
} list_head;

typedef struct MAT_PARAMS_S {
  int N;
  short *A;
  short *B;
  int *C;
} mat_params;

typedef struct CORE_PORTABLE_S {
  u8 portable_id;
} core_portable;

typedef struct RESULTS_S {
  short seed1;
  short seed2;
  short seed3;
  void *memblock[4];
  unsigned size;
  unsigned iterations;
  unsigned execs;
  struct list_head_s *list;
  mat_params mat;
  u16 crc;
  u16 crclist;
  u16 crcmatrix;
  u16 crcstate;
  short err;
  core_portable port;
} core_results;

u16 crc16(short newval, u16 crc);
u16 crcu16(u16 newval, u16 crc);
list_head *core_list_find(list_head *list, list_data *info);
list_head *core_list_reverse(list_head *list);
list_head *core_list_remove(list_head *item);
list_head *core_list_undo_remove(list_head *item_removed,
                                 list_head *item_modified);
typedef int (*list_cmp)(list_data *a, list_data *b, core_results *res);
list_head *core_list_mergesort(list_head *list, list_cmp cmp,
                               core_results *res);
int cmp_idx(list_data *a, list_data *b, core_results *res);
int cmp_complex(list_data *a, list_data *b, core_results *res);

u16 core_bench_list(core_results *res, short finder_idx) {
  u16 retval = 0;
  u16 found = 0, missed = 0;
  list_head *list = res->list;
  short find_num = res->seed3;
  list_head *this_find;
  list_head *finder, *remover;
  list_data info = {0};
  short i;

  info.idx = finder_idx;
  for (i = 0; i < find_num; i++) {
    info.data16 = (i & 0xff);
    this_find = core_list_find(list, &info);
    list = core_list_reverse(list);
    if (this_find == NULL) {
      missed++;
      retval += (list->next->info->data16 >> 8) & 1;
    } else {
      found++;
      if (this_find->info->data16 & 0x1)
        retval += (this_find->info->data16 >> 9) & 1;
      if (this_find->next != NULL) {
        finder = this_find->next;
        this_find->next = finder->next;
        finder->next = list->next;
        list->next = finder;
      }
    }
    if (info.idx >= 0)
      info.idx++;
  }
  retval += found * 4 - missed;
  if (finder_idx > 0)
    list = core_list_mergesort(list, cmp_complex, res);
  remover = core_list_remove(list->next);
  finder = core_list_find(list, &info);
  if (!finder)
    finder = list->next;
  while (finder) {
    retval = crc16(list->info->data16, retval);
    finder = finder->next;
  }
  remover = core_list_undo_remove(remover, list->next);
  list = core_list_mergesort(list, cmp_idx, NULL);
  finder = list->next;
  while (finder) {
    retval = crc16(list->info->data16, retval);
    finder = finder->next;
  }
  return retval;
}
```    
```x86
    .text
    .globl    core_bench_list
    .type   core_bench_list, @function
core_bench_list:
.LFB0:
    endbr64
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $80, %rsp
    movq    %rdi, -72(%rbp)  # Store res (first argument) on stack
    movl    %esi, %eax
    movw    %ax, -76(%rbp)   # Store finder_idx (second argument) on stack

    # Initialize local variables
    movw    $0, -2(%rbp)     # retval = 0
    movw    $0, -4(%rbp)     # found = 0
    movw    $0, -6(%rbp)     # missed = 0

    # list = res->list
    movq    -72(%rbp), %rax
    movq    56(%rax), %rax
    movq    %rax, -16(%rbp)

    # find_num = res->seed3
    movq    -72(%rbp), %rax
    movzwl  4(%rax), %eax
    movw    %ax, -28(%rbp)

    # Initialize info struct
    movl    $0, -52(%rbp)    # info.data16 = 0
    movzwl  -76(%rbp), %eax
    movw    %ax, -50(%rbp)   # info.idx = finder_idx

    # Start of for loop
    movw    $0, -26(%rbp)    # i = 0
    jmp     .L2

.L7:  # Loop body
    # info.data16 = (i & 0xff)
    movzwl  -26(%rbp), %eax
    movzbl  %al, %eax
    movw    %ax, -52(%rbp)

    # this_find = core_list_find(list, &info)
    leaq    -52(%rbp), %rdx
    movq    -16(%rbp), %rax
    movq    %rdx, %rsi
    movq    %rax, %rdi
    call    core_list_find@PLT
    movq    %rax, -48(%rbp)

    # list = core_list_reverse(list)
    movq    -16(%rbp), %rax
    movq    %rax, %rdi
    call    core_list_reverse@PLT
    movq    %rax, -16(%rbp)

    # if (this_find == NULL)
    cmpq    $0, -48(%rbp)
    jne     .L3

    # missed++
    movzwl  -6(%rbp), %eax
    addl    $1, %eax
    movw    %ax, -6(%rbp)

    # retval += (list->next->info->data16 >> 8) & 1
    movq    -16(%rbp), %rax
    movq    (%rax), %rax
    movq    8(%rax), %rax
    movzwl  (%rax), %eax
    sarw    $8, %ax
    andl    $1, %eax
    addw    %ax, -2(%rbp)
    jmp     .L4

.L3:  # else branch
    # found++
    movzwl  -4(%rbp), %eax
    addl    $1, %eax
    movw    %ax, -4(%rbp)

    # if (this_find->info->data16 & 0x1)
    movq    -48(%rbp), %rax
    movq    8(%rax), %rax
    movzwl  (%rax), %eax
    cwtl
    andl    $1, %eax
    testl   %eax, %eax
    je      .L5

    # retval += (this_find->info->data16 >> 9) & 1
    movq    -48(%rbp), %rax
    movq    8(%rax), %rax
    movzwl  (%rax), %eax
    sarw    $9, %ax
    andl    $1, %eax
    addw    %ax, -2(%rbp)

.L5:
    # if (this_find->next != NULL)
    movq    -48(%rbp), %rax
    movq    (%rax), %rax
    testq   %rax, %rax
    je      .L4

    # Rearrange list (finder = this_find->next; ...)
    movq    -48(%rbp), %rax
    movq    (%rax), %rax
    movq    %rax, -24(%rbp)
    movq    -24(%rbp), %rax
    movq    (%rax), %rdx
    movq    -48(%rbp), %rax
    movq    %rdx, (%rax)
    movq    -16(%rbp), %rax
    movq    (%rax), %rdx
    movq    -24(%rbp), %rax
    movq    %rdx, (%rax)
    movq    -16(%rbp), %rax
    movq    -24(%rbp), %rdx
    movq    %rdx, (%rax)

.L4:
    # if (info.idx >= 0) info.idx++
    movzwl  -50(%rbp), %eax
    testw   %ax, %ax
    js      .L6
    movzwl  -50(%rbp), %eax
    addl    $1, %eax
    movw    %ax, -50(%rbp)

.L6:
    # i++
    movzwl  -26(%rbp), %eax
    addl    $1, %eax
    movw    %ax, -26(%rbp)

.L2:  # Loop condition
    movzwl  -26(%rbp), %eax
    cmpw    -28(%rbp), %ax
    jl      .L7

    # retval += found * 4 - missed
    movzwl  -4(%rbp), %eax
    sall    $2, %eax
    subw    -6(%rbp), %ax
    addw    %ax, -2(%rbp)

    # if (finder_idx > 0)
    cmpw    $0, -76(%rbp)
    jle     .L8

    # list = core_list_mergesort(list, cmp_complex, res)
    movq    -72(%rbp), %rdx
    movq    -16(%rbp), %rax
    movq    cmp_complex@GOTPCREL(%rip), %rcx
    movq    %rcx, %rsi
    movq    %rax, %rdi
    call    core_list_mergesort@PLT
    movq    %rax, -16(%rbp)

.L8:
    # remover = core_list_remove(list->next)
    movq    -16(%rbp), %rax
    movq    (%rax), %rax
    movq    %rax, %rdi
    call    core_list_remove@PLT
    movq    %rax, -40(%rbp)

    # finder = core_list_find(list, &info)
    leaq    -52(%rbp), %rdx
    movq    -16(%rbp), %rax
    movq    %rdx, %rsi
    movq    %rax, %rdi
    call    core_list_find@PLT
    movq    %rax, -24(%rbp)

    # if (!finder) finder = list->next
    cmpq    $0, -24(%rbp)
    jne     .L10
    movq    -16(%rbp), %rax
    movq    (%rax), %rax
    movq    %rax, -24(%rbp)
    jmp     .L10

.L11:  # while (finder) loop
    # retval = crc16(list->info->data16, retval)
    movzwl  -2(%rbp), %edx
    movq    -16(%rbp), %rax
    movq    8(%rax), %rax
    movzwl  (%rax), %eax
    cwtl
    movl    %edx, %esi
    movl    %eax, %edi
    call    crc16@PLT
    movw    %ax, -2(%rbp)

    # finder = finder->next
    movq    -24(%rbp), %rax
    movq    (%rax), %rax
    movq    %rax, -24(%rbp)

.L10:
    cmpq    $0, -24(%rbp)
    jne     .L11

    # remover = core_list_undo_remove(remover, list->next)
    movq    -16(%rbp), %rax
    movq    (%rax), %rdx
    movq    -40(%rbp), %rax
    movq    %rdx, %rsi
    movq    %rax, %rdi
    call    core_list_undo_remove@PLT
    movq    %rax, -40(%rbp)

    # list = core_list_mergesort(list, cmp_idx, NULL)
    movq    -16(%rbp), %rax
    movl    $0, %edx
    movq    cmp_idx@GOTPCREL(%rip), %rcx
    movq    %rcx, %rsi
    movq    %rax, %rdi
    call    core_list_mergesort@PLT
    movq    %rax, -16(%rbp)

    # finder = list->next
    movq    -16(%rbp), %rax
    movq    (%rax), %rax
    movq    %rax, -24(%rbp)
    jmp     .L12

.L13:  # while (finder) loop
    # retval = crc16(list->info->data16, retval)
    movzwl  -2(%rbp), %edx
    movq    -16(%rbp), %rax
    movq    8(%rax), %rax
    movzwl  (%rax), %eax
    cwtl
    movl    %edx, %esi
    movl    %eax, %edi
    call    crc16@PLT
    movw    %ax, -2(%rbp)

    # finder = finder->next
    movq    -24(%rbp), %rax
    movq    (%rax), %rax
    movq    %rax, -24(%rbp)

.L12:
    cmpq    $0, -24(%rbp)
    jne     .L13

    # return retval
    movzwl  -2(%rbp), %eax
    leave
    ret
```
""",
    "direct_with_cot": """I want you to act like a compiler that translate C code into x86 assembly. However, I don't want you to do it directly because that's memorizing. I want you to do so by strictly follow my guide and examples.
###Example:
In order to compile the following code into assembly, we need:
1. first analyze the customized structs types and give them correct offset, size and padding, note that each struct follows the largest alignment basic type in its elements.
2. collect all the constants, name their labels with meaningful names.
3. compile the code using the above help messages. generate AT&T syntax x86_64 assembly.
# Input:
```c
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

typedef struct {
    int company_id;
    char company_name[10];
} Company;

typedef struct {
    int person_age;
    char person_name[10];
    long person_id;
    Company *person_company;
} Person;

int main() {
    Company company = {10, "Company A"};
    Person person = {20, "Alice", 10086, &company};
    printf("person name: %s\n", person.person_name);
    printf("person age: %d\n", person.person_age);
    printf("person id: %ld\n", person.person_id);
    printf("person company id: %d\n", person.person_company->company_id);
    printf("person company name: %s\n", person.person_company->company_name);
    person.person_age += 5;
    company.company_id -=5;
    printf("person age: %d\n", person.person_age);
    printf("person company id: %d\n", person.person_company->company_id);

    return 0;
}


```
# Output:
```x86
	.text
# Struct definitions:
# Company {
#     int company_id;        // offset 0, size 4
#     char company_name[10]; // offset 4, size 10
# }; // total size 16 (14 + 2 padding), aligned to 4 bytes

# Person {
#     int person_age;        // offset 0, size 4
#     char person_name[10];  // offset 4, size 10
#     long person_id;        // offset 16, size 8
#     Company *person_company; // offset 24, size 8
# }; // total size 32, aligned to 8 bytes

# String constants
	.section	.rodata
.LC_print_person_name_str:
	.string	"person name: %s\n"
.LC_print_person_age_str:
	.string	"person age: %d\n"
.LC_print_person_id_str:
	.string	"person id: %ld\n"
.LC_print_person_company_id_str:
	.string	"person company id: %d\n"
.LC_print_person_comany_name_str:
	.string	"person company name: %s\n"
.LC_company_name:
	.string "Company A"
.LC_person_name:
	.string "Alice"

	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	endbr64
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$48, %rsp

    # Initialize Company struct
	movl	$10, -16(%rbp)           # company.company_id = 10
	movq	.LC_company_name(%rip), %rax
	movq	%rax, -12(%rbp)          # Copy "Company A" to company.company_name
	movw	$65, -4(%rbp)            # Null-terminate company.company_name

    # Initialize Person struct
	movl	$20, -48(%rbp)           # person.person_age = 20
	movq	.LC_person_name(%rip), %rax
	movq	%rax, -44(%rbp)          # Copy "Alice" to person.person_name
	movw	$0, -36(%rbp)            # Null-terminate person.person_name
	movq	$10086, -32(%rbp)        # person.person_id = 10086
	leaq	-16(%rbp), %rax
	movq	%rax, -24(%rbp)          # person.person_company = &company

    # printf("person name: %s\n", person.person_name);
	leaq	-48(%rbp), %rax
	addq	$4, %rax                 # Calculate address of person.person_name
	movq	%rax, %rsi
	leaq	.LC_print_person_name_str(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT

    # printf("person age: %d\n", person.person_age);
	movl	-48(%rbp), %eax          # Load person.person_age
	movl	%eax, %esi
	leaq	.LC_print_person_age_str(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT

    # printf("person id: %ld\n", person.person_id);
	movq	-32(%rbp), %rax          # Load person.person_id
	movq	%rax, %rsi
	leaq	.LC_print_person_id_str(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT

    # printf("person company id: %d\n", person.person_company->company_id);
	movq	-24(%rbp), %rax          # Load person.person_company
	movl	(%rax), %eax             # Load company_id
	movl	%eax, %esi
	leaq	.LC_print_person_company_id_str(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT

    # printf("person company name: %s\n", person.person_company->company_name);
	movq	-24(%rbp), %rax          # Load person.person_company
	addq	$4, %rax                 # Calculate address of company_name
	movq	%rax, %rsi
	leaq	.LC_print_person_comany_name_str(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT

    # person.person_age += 5;
	movl	-48(%rbp), %eax
	addl	$5, %eax
	movl	%eax, -48(%rbp)

    # company.company_id -= 5;
	movl	-16(%rbp), %eax
	subl	$5, %eax
	movl	%eax, -16(%rbp)

    # printf("person age: %d\n", person.person_age);
	movl	-48(%rbp), %eax          # Load updated person.person_age
	movl	%eax, %esi
	leaq	.LC_print_person_age_str(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT

    # printf("person company id: %d\n", person.person_company->company_id);
	movq	-24(%rbp), %rax          # Load person.person_company
	movl	(%rax), %eax             # Load updated company_id
	movl	%eax, %esi
	leaq	.LC_print_person_company_id_str(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT

    # return 0;
	movl	$0, %eax
	leave
	ret
```
if you understand all above, say a word YES.
""",
    "direct_followup": "now try to compile another code following the above guidelines. make sure you use the proper comments to help you generate.",
    "incremental":"""I want you to act like a compiler that translate C code into x86 assembly. However, I don't want you to do it directly because that's memorizing. I want you to do so by strictly follow my guide and examples.
###Example:
In order to compile the following code into assembly, we need:
1. first analyze the customized structs types and give them correct offset, size and padding, note that each struct follows the largest alignment basic type in its elements.
2. collect all the constants, name their labels with meaningful names.
3. compile the code using the above help messages. generate AT&T syntax x86_64 assembly.
When you generate assembly, i want you to generate it incrementally, you should first generate the control flow, then each implementation in each basicblock.
we will first have an assembly OUTPUT1 that implement the controlflow correctly, then fill each basicblock to get FINALOUTPUT. To do that, we also need the results in 1 and 2.

#Example Input:
```c
#include <stdio.h>

typedef struct {
    int company_id;
    char company_name[10];
} Company;

typedef struct {
    int person_age;
    char person_name[10];
    long person_id;
    Company *person_company;
} Person;

void foo(Person *person1, Person *person2) {
    printf("enter foo\n");
    if(person1->person_age > person2->person_age) {
        person1->person_age += 5;
    } else {
        person2->person_age += 5;
    }
    printf("exit foo\n");
}
```
# Example OUTPUT1:
```x86
	.text
# Struct definitions:
# Company {
#     int company_id;        // offset 0, size 4
#     char company_name[10]; // offset 4, size 10
# }; // total size 16 (14 + 2 padding), aligned to 4 bytes

# Person {
#     int person_age;        // offset 0, size 4
#     char person_name[10];  // offset 4, size 10
#     long person_id;        // offset 16, size 8
#     Company *person_company; // offset 24, size 8
# }; // total size 32, aligned to 8 bytes
    .section    .rodata
.LC_print_enter:
    .string    "enter foo"
.LC_print_exit:
    .string    "exit foo"
    .text
    .globl    foo
    .type    foo, @function
foo:
.L_foo_entry:
	# Function prologue

    # printf("enter foo\n");

    # if(person1->person_age > person2->person_age)
    #COMPARISON, le->else_branch, gt->if_branch
    jle     .L_else_branch

.L_if_branch:
    # person1->person_age += 5;
    jmp     .L_end_if

.L_else_branch:
    # person2->person_age += 5;


.L_end_if:
    # printf("exit foo\n");

    # Function epilogue
    nop
    leave
    ret
```

#FINALOUTPUT:
```x86
	.text
# Struct definitions:
# Company {
#     int company_id;        // offset 0, size 4
#     char company_name[10]; // offset 4, size 10
# }; // total size 16 (14 + 2 padding), aligned to 4 bytes

# Person {
#     int person_age;        // offset 0, size 4
#     char person_name[10];  // offset 4, size 10
#     long person_id;        // offset 16, size 8
#     Company *person_company; // offset 24, size 8
# }; // total size 32, aligned to 8 bytes
    .section    .rodata
.LC_print_enter:
    .string    "enter foo"
.LC_print_exit:
    .string    "exit foo"
    .text
    .globl    foo
    .type    foo, @function
foo:
.L_foo_entry:
	# Function prologue
    endbr64
    pushq    %rbp
    movq    %rsp, %rbp
    subq    $16, %rsp
    movq    %rdi, -8(%rbp)     # Store person1 pointer
    movq    %rsi, -16(%rbp)    # Store person2 pointer

    # printf("enter foo\n");
    leaq    .LC_print_enter(%rip), %rax
    movq    %rax, %rdi
    call    printf@PLT

    # if(person1->person_age > person2->person_age)
    movq    -8(%rbp), %rax     # Load person1 pointer
    movl    (%rax), %edx       # Load person1->person_age
    movq    -16(%rbp), %rax    # Load person2 pointer
    movl    (%rax), %eax       # Load person2->person_age
    #COMPARISON, le->else_branch, gt->if_branch
    cmpl    %eax, %edx
    jle     .L_else_branch

.L_if_branch:
    # person1->person_age += 5;
    movq    -8(%rbp), %rax     # Load person1 pointer
    movl    (%rax), %eax       # Load person1->person_age
    leal    5(%rax), %edx      # Add 5 to person1->person_age
    movq    -8(%rbp), %rax     # Load person1 pointer
    movl    %edx, (%rax)       # Store updated person1->person_age
    jmp     .L_end_if

.L_else_branch:
    # person2->person_age += 5;
    movq    -16(%rbp), %rax    # Load person2 pointer
    movl    (%rax), %eax       # Load person2->person_age
    leal    5(%rax), %edx      # Add 5 to person2->person_age
    movq    -16(%rbp), %rax    # Load person2 pointer
    movl    %edx, (%rax)       # Store updated person2->person_age

.L_end_if:
    # printf("exit foo\n");
    leaq    .LC_print_exit(%rip), %rax
    movq    %rax, %rdi
    call    printf@PLT

    # Function epilogue
    nop
    leave
    ret
```
If you fully understand, say a word YES""",
    "incremental_followup": """now try to compile another code using the above guidelines. You should now generate only until the OUTPUT1, make sure you check carefully on the control flow, and contains annotation comments that reflecting the code snippet that need to be generated later.
###help message: when dealing with switch-case, generate in if-else style. Don't use jump tables.
```c
#include <stdint.h>
#include <stdio.h>
typedef unsigned char u8;

typedef enum CORE_STATE {
  CORE_START = 0,
  CORE_INVALID,
  CORE_S1,
  CORE_S2,
  CORE_INT,
  CORE_FLOAT,
  CORE_EXPONENT,
  CORE_SCIENTIFIC,
  NUM_CORE_STATES
} core_state_e;

u8 ee_isdigit(u8 c);

enum CORE_STATE core_state_transition(u8 **instr, unsigned *transition_count) {
  u8 *str = *instr;
  u8 NEXT_SYMBOL;
  enum CORE_STATE state = CORE_START;
  for (; *str && state != CORE_INVALID; str++) {
    NEXT_SYMBOL = *str;
    if (NEXT_SYMBOL == ',') {
      str++;
      break;
    }
    switch (state) {
    case CORE_START:
      if (ee_isdigit(NEXT_SYMBOL)) {
        state = CORE_INT;
      } else if (NEXT_SYMBOL == '+' || NEXT_SYMBOL == '-') {
        state = CORE_S1;
      } else if (NEXT_SYMBOL == '.') {
        state = CORE_FLOAT;
      } else {
        state = CORE_INVALID;
        transition_count[CORE_INVALID]++;
      }
      transition_count[CORE_START]++;
      break;
    case CORE_S1:
      if (ee_isdigit(NEXT_SYMBOL)) {
        state = CORE_INT;
        transition_count[CORE_S1]++;
      } else if (NEXT_SYMBOL == '.') {
        state = CORE_FLOAT;
        transition_count[CORE_S1]++;
      } else {
        state = CORE_INVALID;
        transition_count[CORE_S1]++;
      }
      break;
    case CORE_INT:
      if (NEXT_SYMBOL == '.') {
        state = CORE_FLOAT;
        transition_count[CORE_INT]++;
      } else if (!ee_isdigit(NEXT_SYMBOL)) {
        state = CORE_INVALID;
        transition_count[CORE_INT]++;
      }
      break;
    case CORE_FLOAT:
      if (NEXT_SYMBOL == 'E' || NEXT_SYMBOL == 'e') {
        state = CORE_S2;
        transition_count[CORE_FLOAT]++;
      } else if (!ee_isdigit(NEXT_SYMBOL)) {
        state = CORE_INVALID;
        transition_count[CORE_FLOAT]++;
      }
      break;
    case CORE_S2:
      if (NEXT_SYMBOL == '+' || NEXT_SYMBOL == '-') {
        state = CORE_EXPONENT;
        transition_count[CORE_S2]++;
      } else {
        state = CORE_INVALID;
        transition_count[CORE_S2]++;
      }
      break;
    case CORE_EXPONENT:
      if (ee_isdigit(NEXT_SYMBOL)) {
        state = CORE_SCIENTIFIC;
        transition_count[CORE_EXPONENT]++;
      } else {
        state = CORE_INVALID;
        transition_count[CORE_EXPONENT]++;
      }
      break;
    case CORE_SCIENTIFIC:
      if (!ee_isdigit(NEXT_SYMBOL)) {
        state = CORE_INVALID;
        transition_count[CORE_INVALID]++;
      }
      break;
    default:
      break;
    }
  }
  *instr = str;
  return state;
}
```""",
}
