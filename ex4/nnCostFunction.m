function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));%25x401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));%10x26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X=[ones(m,1),X];%5000x401
Z2=X*Theta1';%5000x25
A2=sigmoid(Z2);
A2=[ones(m,1),A2];%5000x26
Z3=A2*Theta2';%5000x10
A3=sigmoid(Z3);
% disp(rst)


for i = 1:m
    trueY=zeros(num_labels,1);
    trueY(y(i))=1;
    a3=A3(i,:)';
    delta3=a3-trueY;%10x1

    delta2=Theta2'*delta3;
    z2=Z2(i,:)';
%     size(delta2)
%     size(z2)
    delta2=delta2(2:end);%25x1
    delta2=delta2.*sigmoidGradient(z2);
    %size(delta2)
    

    a2=A2(i,:)';%26x1
    a1=X(i,:)';%401x1
   
    Theta2_grad=Theta2_grad+delta3*a2';%10x1x1x26
    Theta1_grad=Theta1_grad+delta2*a1'; %25x1x1x401
end

Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;


temp1=Theta1;
temp2=Theta2;
temp1(:,1)=zeros(size(temp1,1),1);
temp2(:,1)=zeros(size(temp2,1),1);
Theta1_grad=Theta1_grad+lambda*temp1/m;
Theta2_grad=Theta2_grad+lambda*temp2/m;




for i=1:m
    output=A3(i,:);
    for j=1:num_labels
        if j==y(i)
            J=J-log(output(j));
        else
            J=J-log(1-output(j));
        end
    end
end

J=J/m;

regular=0;
l1=Theta1(:,2:end);
l2=Theta2(:,2:end);
regular=regular+l1(:)'*l1(:);
regular=regular+l2(:)'*l2(:);
regular=regular*lambda/2/m;
J=J+regular;




% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
