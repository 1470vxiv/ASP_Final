% ================== %
%  LMS Time-varying  %
% ================== %
clc
clear 
close all
tic;
load('project_data2024.mat'); 
%% ========================== data 1==============================%%
trainseq1 = trainseq_varying_1;
data1 = data_varying_1;

%% -------------- LMS equalizer -------------- %%
L = 7;                   % 等化器階數
B = 10;
f = zeros(L,50/B);        % 等化器初始化
a = 0.01;                % stepsize

%% ===== training sequence (bits) ===== %%
trainseq_bit = zeros(1,100);
for k = 1:50
    if trainseq1(k) == (1+j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 0;
        trainseq_bit(2+2*(k-1)) = 0;
    elseif trainseq1(k) == (-1+j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 1;
        trainseq_bit(2+2*(k-1)) = 0;
    elseif trainseq1(k) == (-1-j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 1;
        trainseq_bit(2+2*(k-1)) = 1;
    elseif trainseq1(k) == (1-j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 0;
        trainseq_bit(2+2*(k-1)) = 1;
    end
end

%% ===== Training mode ===== %%
recover_trainseq = zeros(500,50);
recover_trainbit = zeros(500,100);
z1 = zeros(500,400);
recover_zbit1 = zeros(500,800);

for b=1:500
    %define block
    e_xn = zeros(L,50);
    block = zeros(L,50/B);

    for n = 1+(b-1)*450:50+(b-1)*450
        if n < L
            xn = [data1(1,n:-1:1) zeros(1,L-n)].';
        else
            xn = [data1(1,n:-1:n-L+1)].';
        end
        y(n) = f(:,floor((n-1)/B+1)).'*xn;
        if real(y(n)) >= 0 && imag(y(n)) >= 0
            recover_trainseq(b,n-(b-1)*450) = (1+1i)/sqrt(2);
        elseif real(y(n)) < 0 && imag(y(n)) > 0
            recover_trainseq(b,n-(b-1)*450) = (-1+1i)/sqrt(2);
        elseif real(y(n)) < 0 && imag(y(n)) < 0
            recover_trainseq(b,n-(b-1)*450) = (-1-1i)/sqrt(2);
        elseif real(y(n)) > 0 && imag(y(n)) < 0
            recover_trainseq(b,n-(b-1)*450) = (1-1i)/sqrt(2);
        end
        e(n) = trainseq1(n-(b-1)*450) - y(n);

        e_xn(:,n) = e(n)*conj(xn);
        if rem(n,B) == 0
            block(:,n/B) = sum(e_xn(:,n-B+1:n),2); %2 is dimension
            f(:,n/B+1) = f(:,n/B) + a*block(:,n/B); %why conj?
        end
    end

    % ----- symbols to bits ----- %
    for k = 1:50
        if recover_trainseq(b,k) == (1+1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 0;
            recover_trainbit(b,2+(k-1)*2) = 0;
        elseif recover_trainseq(b,k) == (-1+1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 1;
            recover_trainbit(b,2+(k-1)*2) = 0;
        elseif recover_trainseq(b,k) == (-1-1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 1;
            recover_trainbit(b,2+(k-1)*2) = 1;
        elseif recover_trainseq(b,k) == (1-1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 0;
            recover_trainbit(b,2+(k-1)*2) = 1;
        end
    end

    %% ===== Decision-Directed mode ===== %%
    %define block
    e_xn = zeros(L,400);
    block = zeros(L,400/B);
    k = 1;

    for i = 50+1+(b-1)*450:450+(b-1)*450
        xn = [data1(i:-1:i-L+1)].';
        y(i) = f(:,floor((i-1)/B+1)).'*xn;
        if real(y(i)) >= 0 && imag(y(i)) >= 0
            z1(b,i-(b-1)*450-50) = (1+j*1)/sqrt(2);
        elseif real(y(i)) < 0 && imag(y(i)) > 0
            z1(b,i-(b-1)*450-50) = (-1+j*1)/sqrt(2);
        elseif real(y(i)) < 0 && imag(y(i)) < 0
            z1(b,i-(b-1)*450-50) = (-1-j*1)/sqrt(2);
        elseif real(y(i))>0 && imag(y(i)) <0
            z1(b,i-(b-1)*450-50) = (1-j*1)/sqrt(2);
        end
        e(i) = z1(b,i-(b-1)*450-50) - y(i);
        e_xn(:,i) = e(i)*conj(xn);
        if rem(i,B) == 0
            block(:,i/B) = sum(e_xn(:,i-B+1:i),2); %2 is dimension
            f(:,i/B+1) = f(:,i/B) + a*block(:,i/B); %why conj?
        end
        k = k+1;
    end

    % ----- symbols to bits ----- %
    for k = 1:400
        if z1(b,k) == (1+1i*1)/sqrt(2)
            recover_zbit1(b,1+(k-1)*2) = 0;
            recover_zbit1(b,2+(k-1)*2) = 0;
        elseif z1(b,k) == (-1+1i*1)/sqrt(2)
            recover_zbit1(b,1+(k-1)*2) = 1;
            recover_zbit1(b,2+(k-1)*2) = 0;
        elseif z1(b,k) == (-1-1i*1)/sqrt(2)
            recover_zbit1(b,1+(k-1)*2) = 1;
            recover_zbit1(b,2+(k-1)*2) = 1;
        elseif z1(b,k) == (1-1i*1)/sqrt(2)
            recover_zbit1(b,1+(k-1)*2) = 0;
            recover_zbit1(b,2+(k-1)*2) = 1;
        end
    end
end
%%
SERtotal1 = 0;
SER1=0;
BERtotal1 = 0;
BER1=0;
for b=1:500
    for k = 1:50
        if recover_trainseq(b,k) ~= trainseq1(k);
            SER1 = SER1+1;
        end
    end
    SERtotal1=SERtotal1+SER1;
    SER1=0;
    for k = 1:100
        if recover_trainbit(b,k) ~= trainseq_bit(k);
            BER1 = BER1+1;
        end
    end
    BERtotal1 = BERtotal1+BER1;
    BER1 = 0;
end

SERtotal1 = SERtotal1/25000
BER1 = BERtotal1/50000

%e = abs(e.^2);
%figure(1)
%plot(1:50,e(1:50));
%title('Time varying case1 (LMS)')
%ylabel('|e(n)^2|');
%ylim([0 2]);

eavg1 = zeros(1,90);
e = abs(e).^2;
for k=1:90
    eavg1(k)=(sum(e(1+5*(k-1):1+5*(k-1)+4)))/5;
end
figure(1)
plot(1:90,eavg1)
ylabel('|e(n)^2|');
xlabel('Iteration');
ylim([0 2]);
title('Time-varying case1 (LMS)');

% % ------ Check plot ------%
figure(2)
%plot(1:length(data1)+1,f);
plot(1:length(data1)/B+1,f(1,:),...
     1:length(data1)/B+1,f(2,:),...
     1:length(data1)/B+1,f(3,:),...
     1:length(data1)/B+1,f(4,:),...
     1:length(data1)/B+1,f(5,:),...
     1:length(data1)/B+1,f(6,:),...
     1:length(data1)/B+1,f(7,:))
legend('l=1','l=2','l=3','l=4','l=5','l=6','l=7');
title('Time-varying case1 - adaptive filter factor (Block LMS)')

figure(3)
plot(1:1000,f(1,1:1000)-f(1,length(f(1,1:1000))));
legend('l=1')
title('Time-varying case1 - adaptive filter f-f* (Block LMS, l=1)');

%% ========================== data 2==============================%%
trainseq2 = trainseq_varying_2;
data2 = data_varying_2;

%% -------------- LMS equalizer -------------- %%
L = 7;                   % 等化器階數
B =10;
f = zeros(L,50/B);        % 等化器初始化
a = 0.01;                % stepsize

%% ===== training sequence (bits) ===== %%
trainseq_bit = zeros(1,100);
for k = 1:50
    if trainseq2(k) == (1+1i)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 0;
        trainseq_bit(2+2*(k-1)) = 0;
    elseif trainseq2(k) == (-1+1i)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 1;
        trainseq_bit(2+2*(k-1)) = 0;
    elseif trainseq2(k) == (-1-1i)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 1;
        trainseq_bit(2+2*(k-1)) = 1;
    elseif trainseq2(k) == (1-1i)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 0;
        trainseq_bit(2+2*(k-1)) = 1;
    end
end

%% ===== Training mode ===== %%
recover_trainseq = zeros(500,50);
recover_trainbit = zeros(500,100);
z2 = zeros(500,400);
recover_zbit2 = zeros(500,800);

for b=1:500
    %define block
    e_xn = zeros(L,200);
    block = zeros(L,200/B);

    for n = 1+(b-1)*450:50+(b-1)*450
        if n < L
            xn = [data2(1,n:-1:1) zeros(1,L-n)].';
        else
            xn = [data2(1,n:-1:n-L+1)].';
        end
        y(n) = f(:,floor((n-1)/B+1)).'*xn;
        if real(y(n)) >= 0 && imag(y(n)) >= 0
            recover_trainseq(b,n-(b-1)*450) = (1+1i)/sqrt(2);
        elseif real(y(n)) < 0 && imag(y(n)) > 0
            recover_trainseq(b,n-(b-1)*450) = (-1+1i)/sqrt(2);
        elseif real(y(n)) < 0 && imag(y(n)) < 0
            recover_trainseq(b,n-(b-1)*450) = (-1-1i)/sqrt(2);
        elseif real(y(n)) > 0 && imag(y(n)) < 0
            recover_trainseq(b,n-(b-1)*450) = (1-1i)/sqrt(2);
        end
        e(n) = trainseq2(n-(b-1)*450) - y(n);
        e_xn(:,n) = e(n)*conj(xn);
        if rem(n,B) == 0
            block(:,n/B) = sum(e_xn(:,n-B+1:n),2); %2 is dimension
            f(:,n/B+1) = f(:,n/B) + a*block(:,n/B); %why conj?
        end
    end

    % ----- symbols to bits ----- %
    for k = 1:50
        if recover_trainseq(b,k) == (1+1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 0;
            recover_trainbit(b,2+(k-1)*2) = 0;
        elseif recover_trainseq(b,k) == (-1+1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 1;
            recover_trainbit(b,2+(k-1)*2) = 0;
        elseif recover_trainseq(b,k) == (-1-1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 1;
            recover_trainbit(b,2+(k-1)*2) = 1;
        elseif recover_trainseq(b,k) == (1-1i*1)/sqrt(2)
            recover_trainbit(b,1+(k-1)*2) = 0;
            recover_trainbit(b,2+(k-1)*2) = 1;
        end
    end

    %% ===== Decision-Directed mode ===== %%
    %define block
    e_xn = zeros(L,1000);
    block = zeros(L,1000/B);
    k = 1;

    for i = 50+1+(b-1)*450:450+(b-1)*450
        xn = [data2(i:-1:i-L+1)].';
        y(i) = f(:,floor((i-1)/B+1)).'*xn;
        if real(y(i)) >= 0 && imag(y(i)) >= 0
            z2(b,i-(b-1)*450-50) = (1+j*1)/sqrt(2);
        elseif real(y(i)) < 0 && imag(y(i)) > 0
            z2(b,i-(b-1)*450-50) = (-1+j*1)/sqrt(2);
        elseif real(y(i)) < 0 && imag(y(i)) < 0
            z2(b,i-(b-1)*450-50) = (-1-j*1)/sqrt(2);
        elseif real(y(i))>0 && imag(y(i)) <0
            z2(b,i-(b-1)*450-50) = (1-j*1)/sqrt(2);
        end
        e(i) = z2(b,i-(b-1)*450-50) - y(i);
        e_xn(:,i) = e(i)*conj(xn);
        if rem(i,B) == 0
            block(:,i/B) = sum(e_xn(:,i-B+1:i),2); %2 is dimension
            f(:,i/B+1) = f(:,i/B) + a*block(:,i/B); %why conj?
        end
        k = k+1;
    end

    % ----- symbols to bits ----- %
    for k = 1:400
        if z2(b,k) == (1+1i*1)/sqrt(2)
            recover_zbit2(b,1+(k-1)*2) = 0;
            recover_zbit2(b,2+(k-1)*2) = 0;
        elseif z2(b,k) == (-1+1i*1)/sqrt(2)
            recover_zbit2(b,1+(k-1)*2) = 1;
            recover_zbit2(b,2+(k-1)*2) = 0;
        elseif z2(b,k) == (-1-1i*1)/sqrt(2)
            recover_zbit2(b,1+(k-1)*2) = 1;
            recover_zbit2(b,2+(k-1)*2) = 1;
        elseif z2(b,k) == (1-1i*1)/sqrt(2)
            recover_zbit2(b,1+(k-1)*2) = 0;
            recover_zbit2(b,2+(k-1)*2) = 1;
        end
    end
end
%%

% error rate
SERtotal2 = 0;
SER2=0;
BERtotal2 = 0;
BER2=0;
for b=1:500
    for k = 1:50
        if recover_trainseq(b,k) ~= trainseq2(k);
            SER2 = SER2+1;
        end
    end
    SERtotal2 = SERtotal2+SER2;
    SER2=0;
    for k = 1:100
        if recover_trainbit(b,k) ~= trainseq_bit(k);
            BER2 = BER2+1;
        end
    end
    BERtotal2 = BERtotal2+BER2;
    BER2 = 0;
end

SERtotal2 = SERtotal2/25000
BER2 = BERtotal2/50000

%e = abs(e.^2);
%figure(3)
%plot(1:50,e(1:50));
%title('Time varying case2 (LMS)')
%ylabel('|e(n)^2|');
%ylim([0 2]);

eavg2 = zeros(1,90);
e = abs(e).^2;
for k=1:90
    eavg2(k)=(sum(e(1+5*(k-1):1+5*(k-1)+4)))/5;
end
figure(4)
plot(1:90,eavg2)
ylabel('|e(n)^2|');
xlabel('Iteration');
ylim([0 2]);
title('Time-varying case2 (Block LMS)')

% % ------ Check plot ------%
figure(5)
%plot(1:length(data1)+1,f);
plot(1:length(data2)/B+1,f(1,:),...
     1:length(data2)/B+1,f(2,:),...
     1:length(data2)/B+1,f(3,:),...
     1:length(data2)/B+1,f(4,:),...
     1:length(data2)/B+1,f(5,:),...
     1:length(data2)/B+1,f(6,:),...
     1:length(data2)/B+1,f(7,:))
legend('l=1','l=2','l=3','l=4','l=5','l=6','l=7');
title('Time-varying case2 - adaptive filter (Block LMS)')

figure(6)
plot(1:1000,f(1,1:1000)-f(1,length(f(1,1:1000))));
legend('l=1')
title('Time-varying case2 - adaptive filter f-f* (Block LMS, l=1)');
%%
ans1= reshape (z1.',1, numel(z1));
ans2= reshape (z2.',1, numel(z2));
ans_BER1 = BER1
ans_SER1 = SERtotal1
ans_BER2 = BER2
ans_SER2 = SERtotal2

save(['ans_varying.mat'],'ans1','ans2');

elapsedTime = toc;  % Stop the timer and get the elapsed time
disp(['Elapsed time: ', num2str(elapsedTime), ' seconds']);








