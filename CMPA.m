%Generate Data
V = linspace(-1.95, 0.7, 200);
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

ideal_diode = Is*(exp((1.2.*V)/0.025)-1);
r_par = Gp.*V;
breakdw = Ib*(exp((-1.2/0.025).*(V + Vb))-1);

I = ideal_diode + r_par - breakdw; 

rand = 0.2 + (-0.2-0.2)*rand(1,200);

Inoise = I + I.*rand;

figure(1);
subplot(4,2,1);
plot(V, I);
hold on;
plot(V, Inoise);
title('Current');
xlabel('V');
ylabel('I');
legend('Current', 'Current with noise');
hold off;

subplot(4,2,2);
semilogy(V, abs(I));
hold on;
semilogy(V, abs(Inoise));
title('Current (semilogy)');
xlabel('V');
ylabel('|I|');
legend('Current', 'Current with noise');
hold off;


% Polynomial fitting

I4 = polyfit(V, I, 4);
I8 = polyfit(V, I, 8);

Inoise4 = polyfit(V, Inoise, 4);
Inoise8 = polyfit(V, Inoise, 8);

subplot(4,2,3);
plot(V, I);
hold on;
plot(V, polyval(I4, V));
hold on;
plot(V, polyval(I8, V));
hold on;
plot(V, Inoise);
hold on;
plot(V, polyval(Inoise4, V));
hold on;
plot(V, polyval(Inoise8, V));
title('Polynomial Fit');
xlabel('V (V)');
ylabel('I (A)');
legend('Current', '4th order fit of current','8th order fit of current', 'Current with noise', '4th order fit of current with noise','8th order fit of current with noise');
hold off;

subplot(4,2,4);
semilogy(V, abs(I));
hold on;
semilogy(V, abs(polyval(I4, V)));
hold on;
semilogy(V, abs(polyval(I8, V)));
hold on;
semilogy(V, abs(Inoise));
hold on;
semilogy(V, abs(polyval(Inoise4, V)));
hold on;
semilogy(V, abs(polyval(Inoise8, V)));
title('Polynomial Filt (semilogy)');
xlabel('V (V)');
ylabel('|I| (A)');
legend('Current', '4th order fit of current','8th order fit of current', 'Current with noise', '4th order fit of current with noise','8th order fit of current with noise');
hold off;

% Nonlinear curve fitting
%a
fo= fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
fo_a = fittype('A.*(exp((1.2.*x)/0.025)-1) + 0.1.*x - C*(exp((-1.2/0.025).*(x + 1.3))-1)');
ff_a = fit(transpose(V),transpose(I),fo_a);
If_a = ff_a(V);

%b
fo_b = fittype('A.*(exp((1.2.*x)/0.025)-1) + B.*x - C*(exp((-1.2/0.025).*(x + 1.3))-1)');
ff_b = fit(transpose(V),transpose(I),fo_b);
If_b = ff_b(V);

%c
fo_c = fittype('A.*(exp((1.2.*x)/0.025)-1) + B.*x - C*(exp((-1.2/0.025).*(x + D))-1)');
ff_c = fit(transpose(V),transpose(I),fo_c);
If_c = ff_c(V);

subplot(4,2,5);
plot(V, I);
hold on;
plot(V, If_a);
hold on;
plot(V, If_b);
hold on;
plot(V, If_c);
title('Nonlinear curve Fitting');
xlabel('V (V)');
ylabel('I (A)');
legend('Current', 'Fit with 2 unknown','Fit with 3 unknown','Fit with 4 unknown');
hold off;

subplot(4,2,6);
semilogy(V, I);
hold on;
semilogy(V, If_a);
hold on;
semilogy(V, If_b);
hold on;
semilogy(V, If_c);
title('Nonlinear curve Fitting (semilogy)');
xlabel('V (V)');
ylabel('|I| (A)');
legend('Current', 'Fit with 2 unknown','Fit with 3 unknown','Fit with 4 unknown');
hold off;

%Neural Net Model
inputs = V;
targets = I;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize); 
net.divideParam.trainRatio = 70/100; 
net.divideParam.valRatio = 15/100; 
net.divideParam.testRatio = 15/100; 
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets); 
performance = perform(net,targets,outputs); 
view(net)
Inn = outputs;


subplot(4,2,7);
plot(V, I);
hold on;
plot(V, Inn);
title('Neural Net Model');
xlabel('V (V)');
ylabel('I (A)');
legend('Current', 'Neural Net Model');
hold off;

subplot(4,2,8);
semilogy(V, I);
hold on;
plot(V, Inn);
title('Neural Net Model (semilogy)');
xlabel('V (V)');
ylabel('I (A)');
legend('Current', 'Neural Net Model');
hold off;