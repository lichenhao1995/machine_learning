//
//  main.c
//  project2s
//
//  Created by apple on 10/24/15.
//  Copyright © 2015 University of Michigan. All rights reserved.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NUMMEMORY 65536 /* maximum number of words in memory */
#define NUMREGS 8 /* number of machine registers */
#define MAXLINELENGTH 1000

typedef struct stateStruct {
    int pc;
    int mem[NUMMEMORY];
    int reg[NUMREGS];
    int memoryAddress;
    int memoryData;
    int instrReg;
    int aluOperand;
    int aluResult;
    int numMemory;
} stateType;

void printState(stateType *, char *);
void run(stateType);
int memoryAccess(stateType *, int);
int convertNum(int);

int
main(int argc, char *argv[])
{
    int i;
    char line[MAXLINELENGTH];
    stateType state;
    FILE *filePtr;
    
    if (argc != 2) {
        printf("error: usage: %s <machine-code file>\n", argv[0]);
        exit(1);
    }
    
    /* initialize memories and registers */
    for (i=0; i<NUMMEMORY; i++) {
        state.mem[i] = 0;
    }
    for (i=0; i<NUMREGS; i++) {
        state.reg[i] = 0;
    }
    
    state.pc=0;
    
    /* read machine-code file into instruction/data memory (starting at
     address 0) */
    
    filePtr = fopen(argv[1], "r");
    if (filePtr == NULL) {
        printf("error: can't open file %s\n", argv[1]);
        perror("fopen");
        exit(1);
    }
    
    for (state.numMemory=0; fgets(line, MAXLINELENGTH, filePtr) != NULL;
         state.numMemory++) {
        if (sscanf(line, "%d", state.mem+state.numMemory) != 1) {
            printf("error in reading address %d\n", state.numMemory);
            exit(1);
        }
        printf("memory[%d]=%d\n", state.numMemory, state.mem[state.numMemory]);
    }
    
    printf("\n");
    
    /* run never returns */
    run(state);
    
    return 0;
}

void
printState(stateType *statePtr, char *stateName)
{
    int i;
    static int cycle = 0;
    printf("\n@@@\nstate %s (cycle %d)\n", stateName, cycle++);
    printf("\tpc %d\n", statePtr->pc);
    printf("\tmemory:\n");
    for (i=0; i<statePtr->numMemory; i++) {
        printf("\t\tmem[ %d ] %d\n", i, statePtr->mem[i]);
    }
    printf("\tregisters:\n");
    for (i=0; i<NUMREGS; i++) {
        printf("\t\treg[ %d ] %d\n", i, statePtr->reg[i]);
    }
    printf("\tinternal registers:\n");
    printf("\t\tmemoryAddress %d\n", statePtr->memoryAddress);
    printf("\t\tmemoryData %d\n", statePtr->memoryData);
    printf("\t\tinstrReg %d\n", statePtr->instrReg);
    printf("\t\taluOperand %d\n", statePtr->aluOperand);
    printf("\t\taluResult %d\n", statePtr->aluResult);
}

/*
 * Access memory:
 *     readFlag=1 ==> read from memory
 *     readFlag=0 ==> write to memory
 * Return 1 if the memory operation was successful, otherwise return 0
 */
int
memoryAccess(stateType *statePtr, int readFlag)
{
    static int lastAddress = -1;
    static int lastReadFlag = 0;
    static int lastData = 0;
    static int delay = 0;
    
    if (statePtr->memoryAddress < 0 || statePtr->memoryAddress >= NUMMEMORY) {
        printf("memory address out of range\n");
        exit(1);
    }
    
    /*
     * If this is a new access, reset the delay clock.
     */
    if ( (statePtr->memoryAddress != lastAddress) ||
        (readFlag != lastReadFlag) ||
        (readFlag == 0 && lastData != statePtr->memoryData) ) {
        delay = statePtr->memoryAddress % 3;
        lastAddress = statePtr->memoryAddress;
        lastReadFlag = readFlag;
        lastData = statePtr->memoryData;
    }
    
    if (delay == 0) {
        /* memory is ready */
        if (readFlag) {
            statePtr->memoryData = statePtr->mem[statePtr->memoryAddress];
        } else {
            statePtr->mem[statePtr->memoryAddress] = statePtr->memoryData;
        }
        return 1;
    } else {
        /* memory is not ready */
        delay--;
        return 0;
    }
}

int
convertNum(int num)
{
    /* convert a 16-bit number into a 32-bit integer */
    if (num & (1 << 15) ) {
        num -= (1 << 16);
    }
    return num;
}

void
run(stateType state)
{
    int bus;
    //int programCounter = 0;
fetch:
    printState(&state, "fetch");
    bus = state.pc;
    state.memoryAddress = bus;
Memory:
    printState(&state, "Memory");
    if (memoryAccess(&state, 1)) {
        goto fetch2;
    }
    else
        goto Memory;
fetch2:
    printState(&state, "fetch2");
    state.pc++;
    bus = state.memoryData;
    state.instrReg = bus;
decode:
    printState(&state, "decode");
    bus=state.reg[(state.instrReg >> 19) & 7];
    state.aluOperand= bus;
    if (state.instrReg>>22 == 0) {
        goto add;
    }
    else if (state.instrReg>>22 == 1) {
        goto nand;
    }
    else if (state.instrReg>>22 == 2) {
        goto lw;
    }
    else if (state.instrReg>>22 == 3) {
        goto sw;
    }
    else if (state.instrReg>>22 == 4) {
        goto beq;
    }
    else if (state.instrReg>>22 == 5) {
        goto jalr;
    }
    else if (state.instrReg>>22 == 6) {
        goto halt;
    }
    else if (state.instrReg>>22 == 7) {
        goto fetch;
    }
add:
    printState(&state, "add");
    bus=state.reg[(state.instrReg >> 16) & 7];
    state.aluResult = state.aluOperand + bus;
add2:
    printState(&state, "add2");
    bus=state.aluResult;
    state.reg[state.instrReg & 7]=bus;
    goto fetch;
nand:
    printState(&state, "nand");
    bus=state.reg[(state.instrReg >> 16) & 7];
    state.aluResult = ~(state.aluOperand & bus);
nand2:
    printState(&state, "nand2");
    bus=state.aluResult;
    state.reg[state.instrReg & 7]=bus;
    goto fetch;
lw:
    printState(&state, "lw");
    bus=convertNum(state.instrReg & 0xFFFF);
    state.aluResult = state.aluOperand + bus;
lw1:
    printState(&state, "lw1");
    bus = state.aluResult;
    state.memoryAddress = bus;
lwMem:
    printState(&state, "lwMem");
    if(memoryAccess(&state, 1)) {
        goto lw2;
    }
    else
        goto lwMem;
lw2:
    printState(&state, "lw2");
    bus=state.memoryData;
    state.reg[(state.instrReg >> 16) & 7]= bus;
    goto fetch;
sw:
    printState(&state, "sw");
    bus=convertNum(state.instrReg & 0xFFFF);
    state.aluResult = state.aluOperand + bus;
sw1:
    printState(&state, "sw1");
    bus = state.aluResult;
    state.memoryAddress = bus;
sw2:
    printState(&state, "sw2");
    bus = state.reg[(state.instrReg>>16) & 7];
    state.memoryData = bus;
swMem:
    printState(&state, "swMem");
    if(memoryAccess(&state, 0)) {
        goto fetch;
    }
    else
        goto swMem;
beq:
    printState(&state, "beq");
    bus = state.reg[(state.instrReg>>16) & 7];
    state.aluResult = state.aluOperand - bus;

beq2:
    printState(&state, "beq2");
    bus = state.pc;
    state.aluOperand=bus;
    if (state.aluResult == 0)
        goto beq3;
    else
        goto fetch;
beq3:
    printState(&state, "beq3");
    bus = convertNum(state.instrReg & 0xFFFF);
    state.aluResult=state.aluOperand+bus;
beq4:
    printState(&state, "beq4");
    bus = state.aluResult;
    state.pc = bus;
    goto fetch;
jalr:
    printState(&state, "jalr");
    bus = state.pc;
    state.reg[(state.instrReg>>16 & 7)] = bus;
jalr2:
    printState(&state, "jalr2");
    bus = state.reg[(state.instrReg>>19 & 7)];
    state.pc = bus;
    goto fetch;
halt:
    printState(&state, "halt");
    exit(0);
}


