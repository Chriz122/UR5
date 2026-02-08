function  plotCalbPointErr(err,rn,cn)
N=rn*cn;
num=length(err);
[x,y]=meshgrid(1:cn,-rn:-1);
x=reshape(x,1,N);
y=reshape(y,1,N);
m=1;
ct=1;
cEr={'-xErr','-yErr','-zErr'};
for n=1:N:num
    %依序繪出校正點的值
    sct=num2str(ct);
    for q=1:3
        subplot(3,3,(m-1)*3+q); 
        plot(x,y,'+');
        title([sct cEr{q}]); 
        for p=1:N
            Er=err(n+p-1,q);
            sEr=num2str(round(Er,3));
%             yErr=num2str(round(err(n+p-1,q),3));
%             zErr=num2str(round(err(n+p-1,q),3));
            if Er<=1
                text(x(p),y(p),sEr,'FontSize',4,'color',[0 0 0])
            elseif Er<=10
                text(x(p),y(p),sEr,'FontSize',6,'color',[0 0 1])
             elseif Er<=20
                text(x(p),y(p),sEr,'FontSize',6,'color',[1 0 0])
            else 
                 text(x(p),y(p),sEr,'FontSize',6,'color',[1 0 1])
            end   
       end
    end
    if uint8(mod(m,3))==0
        m=0;
        figure
    end
    m=m+1;
    ct=ct+1;
end