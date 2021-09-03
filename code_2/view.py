import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from django.shortcuts import render
from django.conf import settings
def get_res(D_o: float):
    teta_HO=[]
    input_teta_txt=open(settings.STATICFILES_DIRS[1]/'input_teta.txt','r+')
    for num in input_teta_txt.readlines():
        teta_HO.append(float(num))
    # D_o=0.324
    t=0.0205
    D_in=D_o-(2*t)
    A_out=math.pi*(D_o**2)/4
    A_in=math.pi*(D_in**2)/4
    Area=A_out-A_in
    I=math.pi*(D_o**4-D_in**4)/64
    ro_s=7850
    ro_c=494
    ro_w=1025
    m_s=Area*ro_s
    m_c=A_in*ro_c
    m_bouy=A_out*ro_w
    m_subm=m_s+m_c-m_bouy
    g=9.81
    Depth=1800
    k=300000
    E=207000000000
    delta_Z=1600
    Z_A=delta_Z-(D_o/2)
    size_teta=len(teta_HO)  # teta_HO should be positive values.
    # creating list with teta size
    teta_HO_rad=[0]*size_teta
    H_T=[0]*size_teta
    X_TDP_cat=[0]*size_teta
    S_TDP_cat=[0]*size_teta
    landa=[0]*size_teta
    K_BLM=[0]*size_teta
    cur_BLM_0=[0]*size_teta
    kesi_f=[0]*size_teta
    s_f=[0]*size_teta
    S_TDP_BLM=[0]*size_teta
    X_TDP_BLM=[0]*size_teta
    x=np.arange(-3500,9601)/10
    size_x=len(x)
    kesi=np.array([[0.0]*size_teta]*size_x)
    cur_BLM=np.array([[0.0]*size_teta]*size_x)
    mom_BLM=np.array([[0.0]*size_teta]*size_x)
    mom_cat=np.array([[0.0]*size_teta]*size_x)
    s=np.array([[0.0]*size_teta]*size_x)
    mom_fin=np.array([[0.0]*size_teta]*size_x)
    Eff_T=np.array([[0.0]*size_teta]*size_x)
    z=np.array([[0.0]*size_teta]*size_x)
    h=np.array([[0.0]*size_teta]*size_x)
    Ext_Force=np.array([[0.0]*size_teta]*size_x)
    Int_Force=np.array([[0.0]*size_teta]*size_x)
    Wall_T=np.array([[0.0]*size_teta]*size_x)
    x_from_ho=np.array([[0.0]*size_teta]*size_x)
    z_from_ho=np.array([[0.0]*size_teta]*size_x)
    Arc_L=np.array([[0.0]*size_teta]*size_x)
    Arc_L_from_TDP=np.array([[0.0]*size_teta]*size_x)
    x_from_TDP=np.array([[0.0]*size_teta]*size_x)
    z_from_TDP=np.array([[0.0]*size_teta]*size_x)
    coor1=np.array([[0.0]*size_teta]*size_x)
    coor2=np.array([[0.0]*size_teta]*size_x)
    coor1_new=np.array([[0.0]*size_teta]*size_x)
    coor2_new=np.array([[0.0]*size_teta]*size_x)
    sigma_0=np.array([[0.0]*size_teta]*size_x)
    sigma_180=np.array([[0.0]*size_teta]*size_x)
    sigma_90=np.array([[0.0]*size_teta]*size_x)
    for jj in range(size_teta):
        teta_HO_rad[jj]=teta_HO[jj]*math.pi/180
        # Cat Eqs:
        H_T[jj]=Z_A*m_subm*g*math.cos(teta_HO_rad[jj])/(1-math.cos(teta_HO_rad[jj]))
        X_TDP_cat[jj]=Z_A*(math.asinh(math.tan(teta_HO_rad[jj])))/(((((math.tan(teta_HO_rad[jj]))**2)+1)**0.5)-1)
        S_TDP_cat[jj]=Z_A*(math.tan(teta_HO_rad[jj]))/(((((math.tan(teta_HO_rad[jj]))**2)+1)**0.5)-1)
        # Boun Eqs:
        landa[jj]=math.sqrt(E*I/H_T[jj])
        K_BLM[jj]=k*landa[jj]**4/(E*I)
        cur_BLM_0[jj]=K_BLM[jj]*m_subm*g/(k*landa[jj]**2)
        kesi_f[jj]=1*((1/(K_BLM[jj]**0.25))-(K_BLM[jj]**0.25))/((2**0.5)+(K_BLM[jj]**0.25))
        s_f[jj]=landa[jj]*kesi_f[jj]
        S_TDP_BLM[jj]=S_TDP_cat[jj]-s_f[jj]
        X_TDP_BLM[jj]=X_TDP_cat[jj]-s_f[jj]
        #
        for ii in range(size_x):
            kesi[ii,jj]=x[ii]/landa[jj]
            if kesi[ii,jj]<=kesi_f[jj]:
                cur_BLM[ii,jj]=cur_BLM_0[jj]*((2**0.5)/((2**0.5)+(K_BLM[jj]**0.25)))*(
                    math.exp((K_BLM[jj]**0.25)*(kesi[ii,jj]-kesi_f[jj])/(2**0.5)))*(
                                   math.cos((K_BLM[jj]**0.25)*(kesi[ii,jj]-kesi_f[jj])/(2**0.5)))
            else:
                cur_BLM[ii,jj]=cur_BLM_0[jj]*(
                        1-(((K_BLM[jj])**0.25)*(math.exp(-(kesi[ii,jj]-kesi_f[jj])))/((2**0.5)+(K_BLM[jj]**0.25))))

            mom_BLM[ii,jj]=E*I*cur_BLM[ii,jj]

            if x[ii]>=0:
                mom_cat[ii,jj]=m_subm*g*E*I/(H_T[jj]*(math.cosh(m_subm*g*x[ii]/H_T[jj]))**2)

            if x[ii]<=0:
                s[ii,jj]=x[ii]
            else:
                s[ii,jj]=H_T[jj]*math.sinh(x[ii]*m_subm*g/H_T[jj])/(m_subm*g)
            # Bend combine of cat Eqs&Bound:
            if x[ii]<0:
                mom_fin[ii,jj]=mom_BLM[ii,jj]/1000
            else:
                if mom_BLM[ii,jj]>mom_cat[ii,jj]:
                    mom_fin[ii,jj]=mom_cat[ii,jj]/1000
                else:
                    mom_fin[ii,jj]=mom_BLM[ii,jj]/1000

            # Eff T:
            if x[ii]<=0:
                Eff_T[ii,jj]=H_T[jj]/1000
            else:
                Eff_T[ii,jj]=math.sqrt(((s[ii,jj]*m_subm*g)**2)+(H_T[jj]**2))/1000

            if not x[ii]<=0:
                z[ii,jj]=H_T[jj]*(math.cosh(m_subm*g*x[ii]/H_T[jj])-1)/(m_subm*g)

            # Wall T:
            h[ii,jj]=Depth-z[ii,jj]-(D_o/2)
            Ext_Force[ii,jj]=ro_w*g*h[ii,jj]*A_out/1000
            Int_Force[ii,jj]=ro_c*g*(h[ii,jj]-Depth+delta_Z)*A_in/1000
            Wall_T[ii,jj]=Eff_T[ii,jj]-Ext_Force[ii,jj]+Int_Force[ii,jj]

            # if coordinate located at the hang-off:
            x_from_ho[ii,jj]=X_TDP_cat[jj]-x[ii]
            z_from_ho[ii,jj]=-(delta_Z-z[ii,jj]-(D_o/2))
            Arc_L[ii,jj]=S_TDP_cat[jj]-s[ii,jj]

            # if coordinate located at the real TDP:
            Arc_L_from_TDP[ii,jj]=S_TDP_BLM[jj]-Arc_L[ii,jj]
            x_from_TDP[ii,jj]=X_TDP_BLM[jj]-x_from_ho[ii,jj]
            z_from_TDP[ii,jj]=-(-Z_A-z_from_ho[ii,jj])

            # if use coordinate based on the AB
            coor1[ii,jj]=x_from_ho[ii,jj]+(-1306.5)
            coor2[ii,jj]=z_from_ho[ii,jj]+1600

            # str(MPa):
            sigma_0[ii,jj]=((Wall_T[ii,jj]/Area)+(mom_fin[ii,jj]*(D_o/2)/I))/1000
            sigma_180[ii,jj]=((Wall_T[ii,jj]/Area)-(mom_fin[ii,jj]*(D_o/2)/I))/1000
            sigma_90[ii,jj]=(Wall_T[ii,jj]/Area)/1000

    # Output 1:
    x0=0
    xf=-1306.5

    for kk in range(size_teta):
        sc0=np.linspace(x0,int(xf*10),int(abs(xf*2)+1))/10  # Common Inteval for sc
        x1=coor1[:,kk]
        z1=coor2[:,kk]

        # Interpolant fitting:
        f_xz=interpolate.interp1d(x1,z1,fill_value="extrapolate")

        if kk==2:
            f_xz_for_sc0_2=f_xz(sc0)

        # f_xz_for_sc0=np.array([[0.0]*size_teta]*size_x)
        # Evaluating in the new time scale:
        # f_xz_for_sc0[:,kk]=feval('f_xz',f_xz,sc0)
        # First main plot:
        coor1_new=np.array([[0.0]*size_teta]*len(sc0))
        coor2_new=np.array([[0.0]*size_teta]*len(sc0))
        coor1_new[:,0]=sc0
        coor2_new[:,kk]=f_xz(sc0)
        plt.legend(str(kk))
        plt.figure(1)
        plt.plot(coor1_new[:,0],coor2_new[:,kk])
        plt.xlabel('H Coor')
        plt.ylabel('V Coor')
        plt.savefig(settings.STATICFILES_DIRS[1]/'fig_1.png')

    # Just to see the comparing plot:    # just for test,
    plt.figure(2)
    plt.plot(coor1[:,2],coor2[:,2],sc0,f_xz_for_sc0_2)
    plt.savefig(settings.STATICFILES_DIRS[1]/'fig_2.png')
    ## Output 2 & 3:
    t0=0
    tf=2333  # the length of riser

    for jj in range(size_teta):
        sc=np.linspace(t0,tf,tf+1)  # Common Inteval for sc
        x=Arc_L[:,jj]  # the horizontal axis is constant for all the following variable
        y1=Eff_T[:,jj]
        y2=Wall_T[:,jj]
        y3=mom_fin[:,jj]
        y4=sigma_0[:,jj]
        y5=sigma_90[:,jj]
        y6=sigma_180[:,jj]

        # Interpolant fitting:
        f1=interpolate.interp1d(x,y1,fill_value="extrapolate")
        f2=interpolate.interp1d(x,y2,fill_value="extrapolate")
        f3=interpolate.interp1d(x,y3,fill_value="extrapolate")
        f4=interpolate.interp1d(x,y4,fill_value="extrapolate")
        f5=interpolate.interp1d(x,y5,fill_value="extrapolate")
        f6=interpolate.interp1d(x,y6,fill_value="extrapolate")

        f1_for_sc=np.array([[0.0]*size_teta]*len(sc))
        f2_for_sc=np.array([[0.0]*size_teta]*len(sc))
        f3_for_sc=np.array([[0.0]*size_teta]*len(sc))
        f4_for_sc=np.array([[0.0]*size_teta]*len(sc))
        f5_for_sc=np.array([[0.0]*size_teta]*len(sc))
        f6_for_sc=np.array([[0.0]*size_teta]*len(sc))

        # Evaluating in the new time scale:
        f1_for_sc[:,jj]=f1(sc)
        f2_for_sc[:,jj]=f2(sc)
        f3_for_sc[:,jj]=f3(sc)
        f4_for_sc[:,jj]=f4(sc)
        f5_for_sc[:,jj]=f5(sc)
        f6_for_sc[:,jj]=f6(sc)

        if jj==2:
            f1_for_sc_2=f1(sc)
            f2_for_sc_2=f2(sc)
            f3_for_sc_2=f3(sc)
            f4_for_sc_2=f4(sc)
            f5_for_sc_2=f5(sc)
            f6_for_sc_2=f6(sc)

        Arc_L_new=np.array([[0.0]*size_teta]*len(sc))
        mom_fin_new=np.array([[0.0]*size_teta]*len(sc))
        sigma_0_new=np.array([[0.0]*size_teta]*len(sc))

        # Final result2
        Arc_L_new[:,0]=sc
        mom_fin_new[:,jj]=f3_for_sc[:,jj]
        sigma_0_new[:,jj]=f4_for_sc[:,jj]

        # Second main plot:
        plt.figure(3)
        plt.plot(Arc_L_new[:,0],mom_fin_new[:,jj])
        plt.xlabel('L')
        plt.ylabel('Mom')
        plt.legend(str(jj))
        plt.savefig(settings.STATICFILES_DIRS[1]/'fig_3.png')
        # Third main plot:
        plt.figure(4)
        plt.plot(Arc_L_new[:,0],sigma_0_new[:,jj])
        plt.xlabel('L')
        plt.ylabel('Sig')
        plt.savefig(settings.STATICFILES_DIRS[1]/'fig_4.png')

    plt.legend(str(jj))
    plt.figure(3)
    plt.legend(range(1,size_teta+1))
    plt.savefig(settings.STATICFILES_DIRS[1]/'fig_3.png')

    plt.legend(str(jj))
    plt.figure(4)
    plt.legend(range(1,size_teta+1))
    plt.savefig(settings.STATICFILES_DIRS[1]/'fig_4.png')

    # Just to see the comparing plots:    # just for test,
    plt.figure(5)
    plt.plot(Arc_L[:,2],sigma_0[:,2],sc,f4_for_sc_2)
    plt.savefig(settings.STATICFILES_DIRS[1]/'fig_5.png')
    ## please consider any ideas for representing nice visualization in plots e.g., colorful, animated, or etc
    # show graphs
    # plt.show()
def home(request):
    messages=[]
    if request.method=='POST':
        if request.POST.get('OK') is not None:
            D=float(request.POST.get('d_value'))
            if D:
                get_res(D)
                os.system('python manage.py collectstatic --noinput')
                return render(request,'results.html')
            else:
                messages.append({
                    'title':'ERROR ! : ',
                    'type':'danger',
                    'body':'Please enter {D} value if you want to calculate new values.'
                })
                messages.append({
                    'title':'INFO ! : ',
                    'type':'primary',
                    'body':'You can create text file containing {θ} values.'
                })
                print(messages)
        elif request.POST.get('SPR') is not None:
            return render(request,'results.html')
    return render(request,'home.html',{'messages':messages})
