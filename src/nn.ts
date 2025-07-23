export class Node {
    id: string;
    inputLinks: Link[] = [];
    bias = 0.1;
    outputLinks: Link[] = [];
    totalInput: number;
    output: number;
    outputDer = 0;
    inputDer = 0;
    accInputDer = 0;
    numAccumulatedDers = 0;
    activation: ActivationFunction;

    constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
        this.id = id;
        this.activation = activation;
        if(initZero) {
            this.bias = 0;
        }
    }

    updateOutput(): number {
        //store total input into the node
        this.totalInput = this.bias;
        for(let j=0; j<this.inputLinks.length; j++) {
            let link = this.inputLinks[j];
            this.totalInput += link.weight * link.source.output;
        }
        this.output = this.activation.output(this.totalInput);
        return this.output;
    }
}

export interface ErrorFunction {
    error : (output: number, target: number) => number;
    der: (output: number, target: number) => number;
}

export interface ActivationFunction {
    output: (input: number) => number;
    der: (input: number) => number;
}

export interface RegularizationFunction {
    output: (weight: number) => number;
    der: (weight: number) => number;
}

export class Errors {
    public static SQUARE: ErrorFunction = {
        error: (output:number, target:number) => 0.5*Math.pow(output-target, 2),
        der: (output:number, target:number) => output - target
    };
}

/*Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
    if(x === Infinity) {
        return 1;
    } else if(x === -Infinity) {
        return -1;
    } else {
        let e2x = Math.exp(2*x);
        return (e2x - 1)/(e2x + 1);
    }
}