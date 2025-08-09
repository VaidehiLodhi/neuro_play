/**
 * A node in a neural network. 
 * (has a state: total input, output, their derivatives) which changes
 * after every forward and back propagation run
 */
export class Node {
    id: string;
    /**list of input links */
    inputLinks: Link[] = [];
    bias = 0.1;
    outputLinks: Link[] = [];
    totalInput: number;
    /**list of output links */
    outputs: Link[] = [];
    output: number;
    /**error derivative with respect to this node's output */
    outputDer = 0;
    /**error derivative with respect to this node's input */
    inputDer = 0;
    /**
     * accumulated error derivative with respect to this node's total input since
     * the last update. this derivative equals dE/db where b is the node's bias term
     */
    accInputDer = 0;
    /**
     * number of accumulated err. derivatives with respect tot he total input since the last update 
     */
    numAccumulatedDers = 0;
    /**
     * activation function that takes total input and returns node's output
     */
    activation: ActivationFunction;

    /**creates a new node with the provided id, activation function and bias*/
    constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
        this.id = id;
        this.activation = activation;
        if(initZero) {
            this.bias = 0;
        }
    }

    /**recumputes the node's output and returns it. */
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


/**
 * an error function and its derivative
 */
export interface ErrorFunction {
    error : (output: number, target: number) => number;
    der: (output: number, target: number) => number;
}

/**
 * an activation function and its derivative
 */
export interface ActivationFunction {
    output: (input: number) => number;
    der: (input: number) => number;
}

/**
 * a regulatization function that computes a penalty cost for a given weight in the network and its derivative
 */
export interface RegularizationFunction {
    output: (weight: number) => number;
    der: (weight: number) => number;
}

/**built in error functions */
export class Errors {
    public static SQUARE: ErrorFunction = {
        error: (output:number, target:number) => 0.5*Math.pow(output-target, 2),
        der: (output:number, target:number) => output - target
    };
}

/** Polyfill for TANH */
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

/** Built in activation functions */
export class Activations {
    public static TANH: ActivationFunction = {
        output: x => (Math as any).tanh(x),
        der: x => {
            let output = Activations.TANH.output(x);
            return 1 - output * output;
        }
    };

    public static RELU: ActivationFunction = {
        output: x => Math.max(0, x),
        der: x => x <= 0 ? 0 : 1
    };

    public static SIGMOID: ActivationFunction = {
        output: x => 1 / (1 + Math.exp(-x)),
        der: x => {
            let output = Activations.SIGMOID.output(x);
            return output * (1-output);
        }
    };

    public static LINEAR: ActivationFunction = {
        output: x => x,
        der: x => 1
    };
}

export class RegularizationFunction {
    public static L1: RegularizationFunction = {
        output: w => Math.abs(w),
        der: w => w<0 ? -1 : (w>0 ? 1 : 0)
    };
    public static L2: RegularizationFunction = {
        output: w => 0.5 * w * w,
        der: w => w
    };
}

/**
 * a link in a neural network
 * each link has a weight and a source and
 * destination node
 * it has an internal state (error derivative with respect to a particular input)
 * which gets updates after a run of back propagation
 */
export class Link {
    id: string;
    source: Node;
    dest: Node;
    weight= Math.random() - 0.5;
    isDead = false;
    /** Error derivative with respect to this weight */
    errorDer = 0;
    /** Accumulater error derivative since the last update */
    accErrorDer = 0;
    /** Number of accumulated derivative since the last update */
    numAccumulatedDers = 0;
    regularization: RegularizationFunction;

    /**
     * constructs a link in the neural network initialized with random weight.
     * @param source the source node
     * @param dest the destination node
     * @param regulaization the regularization function that computes the penalty for this weight. If null, there will be no regularization.
     */

    constructor(source: Node, dest: Node, regularization: RegularizationFunction, initZero?: boolean) {
        this.id = source.id + "" + dest.id;
        this.source = source;
        this.dest = dest;
        this.regularization = regularization;
        if(initZero) {
            this.weight = 0;
        }
    }
}

/**
 * Builds a neural network
 * 
 * @param networkShape the shape of the newtork.
 * eg. [1, 2, 3, 1] means the network will have one input node, 
 * 2 nodes in the first hidden layer, 
 * 3 nodes in second hidden layer
 * and 1 output nodes
 * @param activation the activation function of every hidden node
 * @param outputActivation the activation function for the output node
 * @param regularization the regularization function that computes a penalty 
 * for a given weight (paramenter) in the network. IF null, there will be no regulaiztion
 * @param inputIds List of ids for the input nodes.
 */

export function buildNetwork(
    networkShape: number[],
    activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[],
    initZero ?: boolean
) : Node[][] {

    let numLayer = networkShape.length;
    let id = 1;
    /**list of layers, with each layer being a list of nodes */
    let network : Node[][] = [];
    for(let layerIdx = 0; layerIdx < numLayer; layerIdx++) {
        let isOutputLayer = layerIdx === numLayer - 1;
        let isInputLayer = layerIdx === 0;
        let currentLayer: Node[] = [];
        network.push(currentLayer);
        let numNodes = networkShape[layerIdx];
        for(let i = 0; i < numNodes; i++) {
            let nodeId = id.toString();
            if(isInputLayer) {
                nodeId = inputIds[i];
            } else {
                id++;
            }
            let node = new Node(nodeId,
                isOutputLayer ? outputActivation : activation,
                initZero
            );
            currentLayer.push(node);
            if(layerIdx >= 1) {
                //this means layer is not the input layer
                //add links from nodes in the previous layer to this node
                for(let j = 0; j < network[layerIdx - 1].length; j++ ) {
                    let prevNode = network[layerIdx - 1][j];
                    let link = new Link(prevNode, node, regularization, initZero);
                    prevNode.outputs.push(link);
                    node.inputLinks.push(link);
                }
            }
        }
    }
    return network;
}

/**
 * runs a forward propagation of the provided input network
 * this method modifies the internal state of the network 
 * - the total input and output of each node in the network
 * 
 * @param network the neural network
 * @param inputs the input array, its length should match 
 * the number of input nodes in the network
 * @return the final output of the network
 */

export function forwardProp(network: Node[][], inputs: number[]): number {
    let inputLayer = network[0];
    if(inputs.length !== inputLayer.length) {
        throw new Error("The number of inputs must match the number of nodes in the input layer");
    }
    //update the input layer
    for(let i = 0; i < inputLayer.length; i++) {
        let node = inputLayer[i];
        node.output = inputs[i];
    }
    for(let layerIdx = 1; layerIdx < network.length; layerIdx++) {
        let currentLayer = network[layerIdx];
        //update all nodes in this layer
        for(let i = 0; i < currentLayer.length; i++) {
            let node = currentLayer[i];
            node.updateOutput();
        }
    }
    return network[network.length - 1][0].output;
}

/**
 * runs a backward propagation using the provided target and
 * computed output of the previous call to forward pass
 * this method modifies the internal state of the network
 * - the error derivatives with respect to each node, and each weight in the network
 * @param network the neural network
 * @param target the provided target
 */

export function backProp(network: Node[][], target: number, errorFunc: ErrorFunction): void {
    //the output node is a special case, we use the user-defined error functionfor the derivative
    let outputNode = network[network.length - 1][0];
    outputNode.outputDer = errorFunc.der(outputNode.output, target);

    //go thru the layers backwards
    for(let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
        let currentLayer = network[layerIdx];
        //compute the error derivative of each node with respect to:
        //its total input
        //each of its input weights
        for(let i = 0; i < currentLayer.length; i++) {
            let node = currentLayer[i];
            node.inputDer = node.outputDer * node.activation.der(node.totalInput);
            node.accInputDer += node.inputDer;
            node.numAccumulatedDers++;
        }
        
        //error derivative with respect to each weight
        for(let i = 0; i < currentLayer.length; i++) {
            let node = currentLayer[i];
            for(let j=0; j<node.inputLinks.length; j++) {
                let link = node.inputLinks[j];
                if(link.isDead) {
                    continue;
                }
                link.errorDer = node.inputDer * link.source.output;
                link.accErrorDer += link.errorDer;
                link.numAccumulatedDers++;
            }
        }
        if(layerIdx === 1) {
            continue;
        }
        let prevLayer = network[layerIdx - 1];
        for(let i=0; i<prevLayer.length; i++) {
            let node = prevLayer[i];
            node.outputDer = 0;
            for(let j=0; j<node.outputs.length; j++) {
                let output = node.outputs[j];
                node.outputDer += output.weight * output.dest.inputDer;
            }
        }
    }
}