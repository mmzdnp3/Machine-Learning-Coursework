%Chwan-Hao Tung
%861052182
%11/10/2016
%CS229
%PS 5

function q1( numrep )
    data = load('class2d.ascii','-ascii');
    Xdata = data(:,1:2);
    Ydata = data(:,3);
    numLayers =  [2 3];
    numUnits = [1 5 20];
    lambdas =  [0.1 0.01 0.001];
    for lay = 1:2
        fig = figure('Name',['Layer ',num2str(numLayers(lay))],'position',[0, 0, 1000, 1000]);
        for u = 1:3
            for lam = 1:3
                lambda = lambdas(lam);
                oldloss = 0;
                eta =1;
                w1 = -1 + (1+1)*randn(numUnits(u),size(Xdata,2)+1);
                w2 = -1 + (1+1)*randn(numUnits(u),numUnits(u)+1);
                wf = -1 + (1+1)*randn(1,numUnits(u)+1);
                oldw1 = [];
                oldwf = [];
                oldw2 = [];
                w1min =[];
                w2min =[];
                wfmin =[];
                a1=[];
                a2=[];
                af=[];
                zeta1 = [];
                zeta2 = [];
                iter = 0;
                minloss=inf;
                for k = 1:numrep
                    while 1
                        iter = iter +1
                        f = forwardProp(Xdata,w1,w2,wf,numLayers(lay));
                        backwardPropUpdate(f,numLayers(lay));
                        f = forwardProp(Xdata,w1,w2,wf,numLayers(lay));
                        w12 = w1.^2;
                        w22 = w2.^2;
                        wf2 = wf.^2;
                        loss = sum(-(Ydata.*log(f+1e-32) + (1-Ydata).*log(1-f+1e-32))) + lambda/2 * (sum(w12(:))+sum(wf2(:))+sum(w22(:)));
                        if(loss > oldloss)
                            w1 = oldw1;
                            wf = oldwf;
                            w2 = oldw2;
                            eta = eta *0.5;
                        else
                            eta = eta *1.05;
                            if(abs(oldloss - loss) < oldloss*1e-8)
                                break;
                            end
                        end
                        oldloss = loss;
                    end
                    if loss < minloss
                        minloss = loss;
                        w1min = w1;
                        w2min = w2;
                        wfmin = wf;
                    end
                end
                subplot(3,3,3*(u-1)+lam);
                plotclassifier(Xdata, Ydata, @(X) myclassifier(X, w1min,w2min,wfmin,numLayers(lay)), 0.5,0);
                drawnow;
                title([' Layer:',num2str(numLayers(lay)),' Units:',num2str(numUnits(u)),' Lambda:',num2str(lambda)]);
            end
        end
        print(fig, '-append', '-dpsc2', '-fillpage', 'q1.ps');
    end
    function [sig] = sigmoid(a)
        sig = 1./(1+exp(-a));
    end
    function [f] = forwardProp(inputs,W1,W2,Wf,numLayers)
        inputs = [ones(size(inputs,1),1) inputs];
        a1 = inputs*W1';
        zeta1 = sigmoid(a1);
        if numLayers ==3
            a2 = [ones(size(zeta1,1),1) zeta1]*W2';
            zeta2 = sigmoid(a2);
        end
        if numLayers ==3
            af = [ones(size(zeta2,1),1) zeta2]*Wf';
        else
            af = [ones(size(zeta1,1),1) zeta1]*Wf';
        end
        f = sigmoid(af);
    end
    function backwardPropUpdate(f,numLayers)
        deltf = f - Ydata;
        delt1 = [];
        delt2 = [];
        if numLayers == 3
            dGa = sigmoid(a2).*(1-sigmoid(a2));
            delt2 = deltf*wf;
            delt2 = delt2(:,2:end).*dGa;
        end
        dGa = sigmoid(a1).*(1-sigmoid(a1));
        if numLayers ==3
            delt1 = delt2*w2;
            delt1 = delt1(:,2:end).*dGa;
        else
            delt1 = deltf*wf;
            delt1 = delt1(:,2:end).*dGa;
        end
        oldwf = wf;
        if numLayers == 3
            wf = wf -eta*(deltf'*[ones(size(zeta2,1),1) zeta2] +lambda*wf);
            oldw2 = w2;
            w2 = w2 -eta*(delt2'*[ones(size(zeta1,1),1) zeta1] +lambda*w2);
        else
            wf = wf -eta*(deltf'*[ones(size(zeta1,1),1) zeta1] +lambda*wf);
        end
        oldw1 = w1;
        w1 = w1-eta*(delt1'*[ones(size(Xdata,1),1) Xdata] +lambda*w1);
    end
    
    function [f] = myclassifier(inputs,W1,W2,Wf, numLayers)
        a = [];
        zeta = [];
        inputs = [ones(size(inputs,1),1) inputs];
        a = inputs*W1';
        zeta = 1./(1+exp(-a));
        for i = 2:numLayers-1
            a = [ones(size(inputs,1),1) zeta]*W2';
            zeta = 1./(1+exp(-a));
        end
        a = [ones(size(inputs,1),1) zeta]*Wf'; 
        f = 1./(1+exp(-a));
    end
end