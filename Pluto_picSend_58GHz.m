% 802.11波形+PLUTO空口传输，修改适合5.8GHz载波版本
% 核心：直接指定5.8e9 Hz载波
clear; clc; close all;

%% ===================== 1. 核心配置（5.8GHz空口模式） =====================
% 通道选择：OverTheAir（空口）/GaussianNoise（仿真）/NoImpairments（无损伤）
channel = "OverTheAir"; 

if strcmpi(channel,"OverTheAir")
    deviceName      = "Pluto";               % PLUTO SDR设备
    txCarrierFreq   = 4.12e9;                 % 5.8GHz载波频率（核心！直接指定）
    txGain          = 0;                    % PLUTO发射增益（范围-90~0，-5兼顾功率和线性）
    rxGain          = 60;                    % PLUTO接收增益（范围0~73，28适配5.8G空口）
    % plutoSerial     = '你的PLUTO序列号';     % 替换为实际序列号（sdrinfo('Pluto')获取）
elseif strcmpi(channel,"GaussianNoise")
    SNR = 30;                                % 仿真模式信噪比
end

%% ===================== 2. 分析工具初始化 =====================
% 1. 图像显示窗口
if ~exist('imFig','var') || ~ishandle(imFig)
    imFig = figure('NumberTitle','off','Name','Image Plot','Position',[100 100 800 600]);
else
    clf(imFig);
end

% 2. 频谱分析仪（接收信号功率谱密度）
spectrumScope = spectrumAnalyzer( ...
    'SpectrumType','power-density', ...
    'Title','5.8GHz Received WLAN Signal Spectrum', ...
    'YLabel','Power spectral density (dB/Hz)', ...
    'Position',[69 376 800 450], ...
    'SampleRate',30e6); % 30MHz（20MHz*1.5过采样）

% 3. 64QAM星座图
refQAM = wlanReferenceSymbols('64QAM');
constellation = comm.ConstellationDiagram( ...
    'Title','Equalized 64QAM Symbols (5.8GHz)', ...
    'ShowReferenceConstellation',true, ...
    'ReferenceConstellation',refQAM, ...
    'Position',[878 376 460 460]);

% 4. EVM计算器（误差矢量幅度）
evmCalculator = comm.EVM('AveragingDimensions',[1 2 3],'MaximumEVMOutputPort',true);

% 5. BER计算器（比特误码率）
bitErrorRate = comm.ErrorRate;

%% ===================== 3. 发射机设计 =====================
% 3.1 图像预处理（读取、缩放、转字节流）
fileTx = 'peppers.png';                   % MATLAB内置测试图像
% fileTx = '4.2.06.png'; 

fData = imread(fileTx);                   % 读取图像
scale = 0.2;                              % 缩放因子（减小数据量，适配空口）
origSize = size(fData);
scaledSize = max(floor(scale.*origSize(1:2)),1);
heightIx = min(round(((1:scaledSize(1))-0.5)./scale+0.5),origSize(1));
widthIx = min(round(((1:scaledSize(2))-0.5)./scale+0.5),origSize(2));
fData = fData(heightIx,widthIx,:);        % 图像缩放
imsize = size(fData);
txImage = fData(:);                       % 转列向量

% 显示发射图像
figure(imFig);
subplot(2,1,1); imshow(fData); title('Transmitted Image (5.8GHz)');
subplot(2,1,2); title('Received Image'); set(gca,'Visible','off');

% 3.2 数据分片（拆分为MSDU，符合802.11标准）
msduLength = 2304;                        % 每个MSDU字节数
numMSDUs = ceil(length(txImage)/msduLength);
padZeros = msduLength-mod(length(txImage),msduLength);
txData = [txImage;zeros(padZeros,1)];     % 补零对齐
txDataBits = double(int2bit(txData,8,false)); % 转8bit比特流

% 生成MAC帧（MPDU）和PHY帧（PSDU）
bitsPerOctet = 8;
data = [];
lengthMPDU = 0;
for i = 0:numMSDUs-1
    frameBody = txData(i*msduLength+1:msduLength*(i+1),:);
    cfgMAC = wlanMACFrameConfig('FrameType','Data','SequenceNumber',i);
    [psdu, lengthMPDU] = wlanMACFrame(frameBody,cfgMAC,'OutputFormat','bits');
    data = [data; psdu];
end

% 3.3 生成802.11a基带波形（64QAM，2/3码率）
nonHTcfg = wlanNonHTConfig;               
nonHTcfg.MCS = 6;                         % 64QAM, 2/3码率
nonHTcfg.NumTransmitAntennas = 1;         
chanBW = nonHTcfg.ChannelBandwidth;
nonHTcfg.PSDULength = double(lengthMPDU); 
scramblerInitialization = randi([1 127],numMSDUs,1);
osf = 1.5;                                % 过采样因子（30MHz采样率）
sampleRate = wlanSampleRate(nonHTcfg);    % 标称20MHz

% 生成WLAN波形（含空闲时间）
txWaveform = wlanWaveformGenerator(data,nonHTcfg, ...
    'NumPackets',numMSDUs, ...
    'IdleTime',20e-6, ...
    'ScramblerInitialization',scramblerInitialization, ...
    'OversamplingFactor',osf);

% 3.4 PLUTO发射配置（5.8GHz载波）
sdrTransmitter = [];
if strcmpi(channel,"OverTheAir")
    % 创建PLUTO发射对象
    sdrTransmitter = sdrtx(deviceName);
    sdrTransmitter.BasebandSampleRate = sampleRate*osf; % 30MHz
    sdrTransmitter.CenterFrequency = txCarrierFreq;     % 5.8GHz载波（核心！）
    sdrTransmitter.Gain = txGain;
    
    % 信号缩放（避免RF饱和）
    powerScaleFactor = 0.8;
    txWaveform = txWaveform.*(1/max(abs(txWaveform))*powerScaleFactor);
    
    % 循环发射波形
    fprintf('PLUTO开始5.8GHz发射...\n');
    transmitRepeat(sdrTransmitter,txWaveform);
end

%% ===================== 4. 接收机设计 =====================
% 4.1 PLUTO接收配置（同5.8GHz载波）
rxWaveform = [];
sdrReceiver = [];
if strcmpi(channel,"OverTheAir")
    % 创建PLUTO接收对象
    sdrReceiver = sdrrx(deviceName);
    sdrReceiver.BasebandSampleRate = sampleRate*osf;    % 30MHz
    sdrReceiver.CenterFrequency = txCarrierFreq;        % 5.8GHz载波（和发射端一致）
    sdrReceiver.OutputDataType = 'double';
    sdrReceiver.GainSource = 'Manual';
    sdrReceiver.Gain = rxGain;
    
    % 设置捕获长度（6倍发射波形，确保覆盖所有包）
    sdrReceiver.SamplesPerFrame = 6*length(txWaveform);
    fprintf('PLUTO开始5.8GHz接收...\n');
    rxWaveform = capture(sdrReceiver,sdrReceiver.SamplesPerFrame,'Samples');
elseif strcmpi(channel,"GaussianNoise")
    % 仿真模式：加高斯噪声
    rxWaveform = awgn(txWaveform,SNR,'measured');
else
    % 无损伤模式
    rxWaveform = txWaveform;
end

% 显示接收信号频谱
spectrumScope(rxWaveform);

% 4.2 重采样（30MHz→20MHz，恢复标称采样率）
aStop = 40; 
ofdmInfo = wlanNonHTOFDMInfo('NonHT-Data',nonHTcfg);
SCS = sampleRate/ofdmInfo.FFTLength;
txbw = max(abs(ofdmInfo.ActiveFrequencyIndices))*2*SCS;
[L,M] = rat(1/osf);
maxLM = max([L M]);
R = (sampleRate-txbw)/sampleRate;
TW = 2*R/maxLM;
b = designMultirateFIR(L,M,TW,aStop);
firrc = dsp.FIRRateConverter(L,M,b);
rxWaveform = firrc(rxWaveform);

% 4.3 包检测与解码（限制检测数量，避免无效循环）
rxWaveformLen = size(rxWaveform,1);
searchOffset = 0;
ind = wlanFieldIndices(nonHTcfg);
Ns = ind.LSIG(2)-ind.LSIG(1)+1;
lstfLen = double(ind.LSTF(2));
minPktLen = lstfLen*5;
pktInd = 1;
fineTimingOffset = [];
packetSeq = [];
rxBit = {};
msduList = {};

while (searchOffset+minPktLen) <= rxWaveformLen && pktInd <= numMSDUs
    % 包检测
    pktOffset = wlanPacketDetect(rxWaveform,chanBW,searchOffset,0.5);
    pktOffset = searchOffset + pktOffset;
    
    if isempty(pktOffset) || (pktOffset+double(ind.LSIG(2))>rxWaveformLen)
        if pktInd == 1
            disp('** 未检测到任何数据包 **');
        end
        break;
    end
    
    fprintf('\n检测到第%d个包，起始索引：%d\n',pktInd,pktOffset+1);
    
    % 粗频偏估计与校正
    nonHT = rxWaveform(pktOffset+(ind.LSTF(1):ind.LSIG(2)),:);
    coarseFreqOffset = wlanCoarseCFOEstimate(nonHT,chanBW);
    nonHT = frequencyOffset(nonHT,sampleRate,-coarseFreqOffset);
    
    % 符号定时同步
    fineTimingOffset = wlanSymbolTimingEstimate(nonHT,chanBW);
    pktOffset = pktOffset + fineTimingOffset;
    
    if (pktOffset<0) || ((pktOffset+minPktLen)>rxWaveformLen)
        searchOffset = pktOffset + 1.5*lstfLen;
        continue;
    end
    
    % 精频偏估计与校正
    nonHT = rxWaveform(pktOffset+(1:7*Ns),:);
    nonHT = frequencyOffset(nonHT,sampleRate,-coarseFreqOffset);
    lltf = nonHT(ind.LLTF(1):ind.LLTF(2),:);
    fineFreqOffset = wlanFineCFOEstimate(lltf,chanBW);
    fineFreqOffset = fineFreqOffset * 0.95; % 二次精校
    nonHT = frequencyOffset(nonHT,sampleRate,-fineFreqOffset);
    cfoCorrection = coarseFreqOffset + fineFreqOffset;
    
    % 信道估计
    demodLLTF = wlanLLTFDemodulate(lltf,chanBW);
    chanEstLLTF = wlanLLTFChannelEstimate(demodLLTF,chanBW);
    noiseVarNonHT = wlanLLTFNoiseEstimate(demodLLTF);
    
    % 包格式检测
    format = wlanFormatDetect(nonHT(ind.LLTF(2)+(1:3*Ns),:), ...
        chanEstLLTF,noiseVarNonHT,chanBW);
    disp(['  检测到包格式：' format]);
    
    if ~strcmp(format,'Non-HT')
        fprintf('  非Non-HT格式，跳过\n');
        searchOffset = pktOffset + 1.5*lstfLen;
        continue;
    end
    
    % L-SIG解码
    [recLSIGBits,failCheck] = wlanLSIGRecover( ...
        nonHT(ind.LSIG(1):ind.LSIG(2),:), ...
        chanEstLLTF,noiseVarNonHT,chanBW);
    
    if failCheck
        fprintf('  L-SIG校验失败，跳过\n');
        searchOffset = pktOffset + 1.5*lstfLen;
        continue;
    else
        fprintf('  L-SIG校验通过\n');
    end
    
    % 解析L-SIG参数（手动计算包采样数）
    [lsigMCS,lsigLen,rxSamples] = helperInterpretLSIG(recLSIGBits,sampleRate,nonHTcfg);
    
    if (rxSamples + pktOffset) > length(rxWaveform)
        disp('** 采样点不足，停止解码 **');
        break;
    end
    
    % 整包频偏校正
    rxWaveform(pktOffset+(1:rxSamples),:) = frequencyOffset(...
        rxWaveform(pktOffset+(1:rxSamples),:),sampleRate,-cfoCorrection);
    
    % 数据域解码
    rxNonHTcfg = nonHTcfg;
    indNonHTData = wlanFieldIndices(rxNonHTcfg,'NonHT-Data');
    [rxPSDU,eqSym] = wlanNonHTDataRecover(rxWaveform(pktOffset+...
        (indNonHTData(1):indNonHTData(2)),:), ...
        chanEstLLTF,noiseVarNonHT,rxNonHTcfg);
    
    % 显示星座图
    constellation(reshape(eqSym,[],1));
    
    % EVM计算
    refSym = wlanClosestReferenceSymbol(eqSym,rxNonHTcfg);
    [evm.RMS,evm.Peak] = evmCalculator(refSym,eqSym);
    fprintf('  EVM峰值：%.3f%%，EVM均方根：%.3f%%\n',evm.Peak,evm.RMS);
    
    % MPDU解码（提取MSDU）
    [cfgMACRx,msduList{pktInd},status] = wlanMPDUDecode(rxPSDU,rxNonHTcfg);
    
    if strcmp(status,'Success')
        disp('  MAC FCS校验通过');
        packetSeq(pktInd) = cfgMACRx.SequenceNumber;
        rxBit{pktInd} = int2bit(hex2dec(cell2mat(msduList{pktInd})),8,false);
    else
        disp('  MAC FCS校验失败，强制提取数据');
        % 强制提取MSDU（兜底逻辑）
        macHeaderBitsLength = 24*bitsPerOctet;
        fcsBitsLength = 4*bitsPerOctet;
        if length(rxPSDU) > (macHeaderBitsLength + fcsBitsLength)
            msduBits = rxPSDU(macHeaderBitsLength+1:end-fcsBitsLength);
        else
            msduBits = zeros(msduLength*8,1);
        end
        % 提取序列号
        if length(rxPSDU) >= 25*bitsPerOctet
            sequenceNumStartIndex = 23*bitsPerOctet+1;
            sequenceNumEndIndex = 25*bitsPerOctet-4;
            conversionLength = sequenceNumEndIndex - sequenceNumStartIndex + 1;
            packetSeq(pktInd) = bit2int(rxPSDU(sequenceNumStartIndex:sequenceNumEndIndex),conversionLength,false);
        else
            packetSeq(pktInd) = pktInd-1;
        end
        % 长度对齐
        if length(msduBits) < msduLength*8
            msduBits = [msduBits; zeros(msduLength*8 - length(msduBits),1)];
        else
            msduBits = msduBits(1:msduLength*8);
        end
        rxBit{pktInd} = double(msduBits);
    end
    
    % 去重包
    if length(unique(packetSeq)) < length(packetSeq)
        rxBit = rxBit(1:length(unique(packetSeq)));
        packetSeq = packetSeq(1:length(unique(packetSeq)));
        break;
    end
    
    % 更新搜索偏移
    searchOffset = pktOffset + double(indNonHTData(2));
    pktInd = pktInd + 1;
end

% 释放PLUTO资源
if strcmpi(channel,"OverTheAir") && ~isempty(sdrTransmitter) && ~isempty(sdrReceiver)
    release(sdrTransmitter);
    release(sdrReceiver);
    fprintf('PLUTO 5.8GHz收发资源已释放\n');
end

%% ===================== 5. 图像重建（增加空值保护） =====================
if ~(isempty(fineTimingOffset) || isempty(pktOffset))
    if ~isempty(rxBit)
        rxData = cat(1,rxBit{:});
        rxData = rxData(1:end-(mod(length(rxData),msduLength*8)));
        rxData = reshape(rxData,msduLength*8,[]);
        
        % 去重包
        if length(packetSeq) > numMSDUs
            numDupPackets = size(rxData,2) - numMSDUs;
            rxData = rxData(:,1:end-numDupPackets);
        end
        
        % 按序列号排序
        if any(packetSeq < numMSDUs) && ~isempty(packetSeq)
            startSeq = [];
            i = -1;
            while isempty(startSeq) && i < numMSDUs
                i = i + 1;
                startSeq = find(packetSeq == i);
            end
            if ~isempty(startSeq)
                rxData = circshift(rxData,[0 -(startSeq(1)-i-1)]);
                
                % 计算BER
                if ~isempty(rxData) && ~isempty(txDataBits)
                    compareLen = min(length(reshape(rxData,[],1)), length(txDataBits));
                    if compareLen > 0
                        err = bitErrorRate(double(rxData(:)),txDataBits(1:compareLen));
                        fprintf('\n5.8GHz传输比特误码率（BER）：\n');
                        fprintf('  BER = %.5f\n  误码数 = %d\n  总传输比特数 = %d\n',err(1),err(2),compareLen);
                    end
                end
            end
        end
        
        % 比特流转图像
        if ~isempty(rxData)
            rxDataFlat = reshape(rxData(:),8,[]);
            if ~isempty(rxDataFlat)
                decData = bit2int(rxDataFlat,8,false)';
                if length(decData) < length(txImage)
                    numMissingData = length(txImage) - length(decData);
                    decData = [decData;NaN(numMissingData,1)];
                else
                    decData = decData(1:length(txImage));
                end
                
                % 显示接收图像
                receivedImage = uint8(reshape(decData,imsize));
                figure(imFig); subplot(2,1,2);
                imshow(receivedImage); title('Received Image (5.8GHz)');
                set(gca,'Visible','on');
                fprintf('\n5.8GHz图像重建完成！\n');
            else
                fprintf('\nrxDataFlat为空，无法重建图像\n');
            end
        else
            fprintf('\nrxData为空，无法重建图像\n');
        end
    else
        fprintf('\nrxBit为空，无法重建图像\n');
    end
else
    fprintf('\n未检测到有效数据包，无法重建图像\n');
end

%% ===================== 6. 释放分析工具资源 =====================
release(spectrumScope);
release(constellation);
release(evmCalculator);
release(bitErrorRate);

%% ===================== 辅助函数：解析L-SIG并计算包采样数 =====================
function [lsigMCS, lsigLen, rxSamples] = helperInterpretLSIG(lsigBits, sampleRate, nonHTcfg)
    % 提取速率比特（前4bit）
    rateBits = lsigBits(1:4);
    rateVal = double(bit2int(rateBits,4,false));
    if rateVal >=0 && rateVal <=7
        lsigMCS = double(rateVal);
    else
        lsigMCS = double(nonHTcfg.MCS);
    end
    
    % 提取长度比特（4-11bit）
    lenBits = lsigBits(5:12);
    lsigLen = double(bit2int(lenBits,8,false));
    if lsigLen == 0
        lsigLen = double(nonHTcfg.PSDULength);
    end
    
    % 手动计算包总采样数
    rxNonHTcfg = wlanNonHTConfig('MCS',lsigMCS,'PSDULength',lsigLen);
    rxSamples = double(calculateWLANPacketSamples(rxNonHTcfg, sampleRate));
end

function totalSamples = calculateWLANPacketSamples(nonHTcfg, sampleRate)
    % 802.11a Non-HT帧结构：LSTF + LLTF + LSIG + Data
    lstfSymbols = 16;
    lltfSymbols = 64;
    lsigSymbols = 32;
    dataSymbols = calculateDataSymbols(nonHTcfg);
    
    samplesPerSymbol = sampleRate * 4e-6; % 20MHz→80 samples/symbol
    totalSamples = (lstfSymbols + lltfSymbols + lsigSymbols + dataSymbols) * samplesPerSymbol;
    totalSamples = round(totalSamples);
end

function dataSymbols = calculateDataSymbols(nonHTcfg)
    % 计算Data字段符号数
    mcs = nonHTcfg.MCS;
    psduLength = nonHTcfg.PSDULength;
    
    mcsMap = {
        0,    1/2,    1;
        1,    1/2,    2;
        2,    3/4,    2;
        3,    1/2,    4;
        4,    3/4,    4;
        5,    1/2,    6;
        6,    2/3,    6;
        7,    3/4,    6;
    };
    
    codeRate = mcsMap{mcs+1,2};
    bitsPerSymbol = mcsMap{mcs+1,3};
    effectiveBitsPerSymbol = 48 * bitsPerSymbol * codeRate;
    
    totalDataBits = psduLength + 16 + 6;
    paddingBits = mod(effectiveBitsPerSymbol - mod(totalDataBits, effectiveBitsPerSymbol), effectiveBitsPerSymbol);
    totalDataBits = totalDataBits + paddingBits;
    
    dataSymbols = ceil(totalDataBits / effectiveBitsPerSymbol);
end

% plot_my_constellation(eqSym);
