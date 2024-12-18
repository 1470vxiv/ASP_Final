clc
clear 
close all

load('project_data2024.mat'); 
trainseq1 = trainseq_static_1; %desired signal d(n)
data1 = data_static_1;

%% -------------- LMS equalizer -------------- %%
L = 7;                  % 等化器階數
f = zeros(L,1000);      % 等化器初始化
a = 0.01;                % stepsize

%% ===== training sequence (bits) ===== %%
trainseq_bit = zeros(1,2000);
recover_trainseq = zeros(1,1000);
for k = 1:1000
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
for n = 1:1000
    if n < L
        xn = [data1(1,n:-1:1) zeros(1,L-n)].';
    else
        xn = [data1(1,n:-1:n-L+1)].';
    end
    y(n) = f(:,n).'*xn;
    if real(y(n)) >= 0 && imag(y(n)) >= 0
        recover_trainseq(n) = (1+j)/sqrt(2);
    elseif real(y(n)) < 0 && imag(y(n)) > 0
        recover_trainseq(n) = (-1+j)/sqrt(2);
    elseif real(y(n)) < 0 && imag(y(n)) < 0
        recover_trainseq(n) = (-1-j)/sqrt(2);
    else real(y(n)) > 0 && imag(y(n)) < 0
        recover_trainseq(n) = (1-j)/sqrt(2);
    end
    
    e(n) = trainseq1(n) - y(n);
    f(:,n+1) = f(:,n) + a*e(n)*conj(xn); %why conj?
end

% ----- symbols to bits ----- %
recover_trainbit = zeros(1,2000);
for k = 1:1000 
    if recover_trainseq(k) == (1+j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 0;
        recover_trainbit(2+(k-1)*2) = 0;
    elseif recover_trainseq(k) == (-1+j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 1;
        recover_trainbit(2+(k-1)*2) = 0;
    elseif recover_trainseq(k) == (-1-j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 1;
        recover_trainbit(2+(k-1)*2) = 1;
    elseif recover_trainseq(k) == (1-j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 0;
        recover_trainbit(2+(k-1)*2) = 1;
    end
end

SER1 = 0;
for k = 1:1000
    if recover_trainseq(k) ~= trainseq1(k);
        SER1 = SER1+1;
    end
end

BER1 = (sum(trainseq_bit~=recover_trainbit))/2000
SER1 = SER1/1000

%% ===== Decision-Directed mode ===== %%
z1 = zeros(1,200000);
k = 1;
for i = 1000+1:length(data1)
    xn = [data1(i:-1:i-L+1)].';
    y(i) = f(:,i).'*xn;
    if real(y(i)) >= 0 && imag(y(i)) >= 0
        z1(k) = (1+j*1)/sqrt(2);
    elseif real(y(i)) < 0 && imag(y(i)) > 0
        z1(k) = (-1+j*1)/sqrt(2);
    elseif real(y(i)) < 0 && imag(y(i)) < 0
        z1(k) = (-1-j*1)/sqrt(2);
    elseif real(y(i)) > 0 && imag(y(i)) < 0
        z1(k) = (1-j*1)/sqrt(2);
    end
    e(i) = z1(k) - y(i);
    f(:,i+1) = f(:,i) + a*e(i)*conj(xn);
    k = k+1;
end

% ----- symbols to bits ----- %
recover_zbit1 = zeros(1,400000);
for k = 1:200000 
    if z1(k) == (1+j*1)/sqrt(2)
        recover_zbit1(1+(k-1)*2) = 0;
        recover_zbit1(2+(k-1)*2) = 0;
    elseif z1(k) == (-1+j*1)/sqrt(2)
        recover_zbit1(1+(k-1)*2) = 1;
        recover_zbit1(2+(k-1)*2) = 0;
    elseif z1(k) == (-1-j*1)/sqrt(2)
        recover_zbit1(1+(k-1)*2) = 1;
        recover_zbit1(2+(k-1)*2) = 1;
    elseif z1(k) == (1-j*1)/sqrt(2)
        recover_zbit1(1+(k-1)*2) = 0;
        recover_zbit1(2+(k-1)*2) = 1;
    end
end

%e = abs(e.^2);
%figure(1)
%plot(1:1000,e(1:1000));
%title('Static case1 (LMS)')
%ylabel('|e(n)^2|');
%ylim([0 2]);

eavg1=zeros(1,200);
e = abs(e).^2;
for k=1:200
    eavg1(k)=(sum(e(1+5*(k-1):1+5*(k-1)+4)))/5;
end
figure(1)
plot(1:200,eavg1)
ylabel('|e(n)^2|');
ylim([0 2]);
title('Static case1 (LMS)')

% % ------ Check plot ------%
figure(2)
plot(1:length(data1)+1,f(1,:),...
    1:length(data1)+1,f(2,:),...
    1:length(data1)+1,f(3,:),...
    1:length(data1)+1,f(4,:),...
    1:length(data1)+1,f(5,:),...
    1:length(data1)+1,f(6,:),...
    1:length(data1)+1,f(7,:))
legend('L=1','L=2','L=3','L=4','L=5','L=6','L=7');
title('Quasi-static case1 - adaptive filter (LMS)');
%% ===== data 2 ===== %%
trainseq2 = trainseq_static_2;
data2 = data_static_2;

%% -------------- LMS equalizer -------------- %%
L = 3;                  % 等化器階數
f = zeros(L,1000);      % 等化器初始化
a = 0.01;                % stepsize

%% ===== training sequence (bits) ===== %%
trainseq_bit = zeros(1,2000);
recover_trainseq = zeros(1,1000);
for k = 1:1000
    if trainseq2(k) == (1+j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 0;
        trainseq_bit(2+2*(k-1)) = 0;
    elseif trainseq2(k) == (-1+j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 1;
        trainseq_bit(2+2*(k-1)) = 0;
    elseif trainseq2(k) == (-1-j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 1;
        trainseq_bit(2+2*(k-1)) = 1;
    else trainseq2(k) == (1-j)/sqrt(2)
        trainseq_bit(1+2*(k-1)) = 0;
        trainseq_bit(2+2*(k-1)) = 1;
    end
end

%% ===== Training mode ===== %%
mm = 1;
for n = 1:1000
    if n < L
        xn = [data2(1,n:-1:1) zeros(1,L-n)].';
    else
        xn = [data2(1,n:-1:n-L+1)].';
    end
    y(n) = f(:,n).'*xn;
    if real(y(n)) >= 0 && imag(y(n)) >= 0
        recover_trainseq(mm) = (1+j)/sqrt(2);
    elseif real(y(n)) < 0 && imag(y(n)) > 0
        recover_trainseq(mm) = (-1+j)/sqrt(2);
    elseif real(y(n)) < 0 && imag(y(n)) < 0
        recover_trainseq(mm) = (-1-j)/sqrt(2);
    else real(y(n)) > 0 && imag(y(n)) < 0
        recover_trainseq(mm) = (1-j)/sqrt(2);
    end
    mm = mm+1;
    
    e(n) = trainseq2(n) - y(n);
    f(:,n+1) = f(:,n) + a*e(n)*conj(xn);
end

% ----- symbols to bits ----- %
recover_trainbit = zeros(1,2000);
for k = 1:1000 
    if recover_trainseq(k) == (1+j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 0;
        recover_trainbit(2+(k-1)*2) = 0;
    elseif recover_trainseq(k) == (-1+j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 1;
        recover_trainbit(2+(k-1)*2) = 0;
    elseif recover_trainseq(k) == (-1-j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 1;
        recover_trainbit(2+(k-1)*2) = 1;
    else recover_trainseq(k) == (1-j*1)/sqrt(2)
        recover_trainbit(1+(k-1)*2) = 0;
        recover_trainbit(2+(k-1)*2) = 1;
    end
end

SER2 = 0;
for k = 1:1000
    if recover_trainseq(k) ~= trainseq2(k);
        SER2 = SER2+1;
    end
end

BER2 = (sum(trainseq_bit~=recover_trainbit))/2000
SER2 = SER2/1000

%% ===== Decision-Directed mode ===== %%
z2 = zeros(1,200000);
k = 1;
for i = 1000+1:length(data2)
    xn = [data2(i:-1:i-L+1)].';
    y(i) = f(:,i).'*xn;
    if real(y(i)) >= 0 && imag(y(i)) >= 0
        z2(k) = (1+j*1)/sqrt(2);
    elseif real(y(i)) < 0 && imag(y(i)) > 0
        z2(k) = (-1+j*1)/sqrt(2);
    elseif real(y(i)) < 0 && imag(y(i)) < 0
        z2(k) = (-1-j*1)/sqrt(2);
    elseif real(y(i))>0 && imag(y(i)) <0
        z2(k) = (1-j*1)/sqrt(2);
    end
    e(i) = z2(k) - y(i);
    f(:,i+1) = f(:,i) + a*e(i)*conj(xn);
    k = k+1;
end

% ----- symbols to bits ----- %
recover_zbit2 = zeros(1,400000);
for k = 1:200000 
    if z2(k) == (1+j*1)/sqrt(2)
        recover_zbit2(1+(k-1)*2) = 0;
        recover_zbit2(2+(k-1)*2) = 0;
    elseif z2(k) == (-1+j*1)/sqrt(2)
        recover_zbit2(1+(k-1)*2) = 1;
        recover_zbit2(2+(k-1)*2) = 0;
    elseif z2(k) == (-1-j*1)/sqrt(2)
        recover_zbit2(1+(k-1)*2) = 1;
        recover_zbit2(2+(k-1)*2) = 1;
    else z2(k) == (1-j*1)/sqrt(2)
        recover_zbit2(1+(k-1)*2) = 0;
        recover_zbit2(2+(k-1)*2) = 1;
    end
end

%e = abs(e.^2);
%figure(3)
%plot(1:1000,e(1:1000));
%title('Static case2 (LMS)')
%ylabel('|e(n)^2|');
%ylim([0 2]);
%%
eavg2=zeros(1,200);
e = abs(e).^2;
for k=1:200
    eavg2(k)=(sum(e(1+5*(k-1):1+5*(k-1)+4)))/5;
end
figure(3)
plot(1:200,eavg2)
ylabel('|e(n)^2|');
ylim([0 2]);
title('Static case2 (LMS)')

% % ------ Check plot ------%
figure(4)
plot(1:length(data2)+1,f(1,:),...
    1:length(data2)+1,f(2,:),...
    1:length(data2)+1,f(3,:))
legend('L=1','L=2','L=3');
title('Static case2 - adaptive filter (LMS)');

ans_BER1 = BER1
ans_SER1 = SER1
ans_BER2 = BER2
ans_SER2 = SER2

save(['ans_static.mat'],'z1','z2');
