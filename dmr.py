import numpy as np
import plotly.graph_objects as go

from scipy.optimize import newton, root
from scipy.signal import find_peaks
from scipy import linalg as la

class Material:
    def __init__(self, rho, E):
        self.rho = rho
        self.E = E

class FuncaoForma:
    def __init__(self, L):
        self.L = L
        self.c0 = np.pi / L

    def f(self, pos):
        return np.sin(self._x(pos))

    def g(self, pos):
        return self.c0 * np.cos(self._x(pos))
    
    def h(self, pos):
        return - (self.c0 ** 2) * np.sin(self._x(pos))
    
    def _x(self, z):
        return self.c0 * z
    
    def intg_f2(self,):
        # x / 2 - sin(2 c0 x) / (4 c0)
        up = self.L / 2 - (np.sin(2 * self._x(self.L)) / (4 * self.c0))
        lo = - (np.sin(2 * self._x(0)) / (4 * self.c0))
        return up - lo

    def intg_g2(self,):
        # x / 2 + sin(2 c0 x) / (4 c0)
        up = self.L / 2 + (np.sin(2 * self._x(self.L)) / (4 * self.c0))
        lo = (np.sin(2 * self._x(0)) / (4 * self.c0))
        return (self.c0 ** 2) * (up - lo)

    def intg_h2(self,):
        # x / 2 - sin(2 c0 x) / (4 c0)
        up = self.L / 2 - (np.sin(2 * self._x(self.L)) / (4 * self.c0))
        lo = - (np.sin(2 * self._x(0)) / (4 * self.c0))
        return (self.c0 ** 4) * (up - lo)
    
    def plot(self, n_points=20):
        fig = go.Figure()
        length = np.linspace(0, self.L, num=n_points)
        forma = np.zeros(len(length))
        for i, z in enumerate(length):
            forma[i] = self.f(z)

        fig.add_trace(go.Scatter(
            x=length,
            y=forma,
            name="Função de Forma",
            mode="lines",
            line=dict(
                color="#000000"
            )
        ))

        return fig
    
class Eixo:
    def __init__(self, R, L, material:Material, r=None, F0=None):
        self.R = R
        self.L = L
        self.r = 0 if r is None else r
        self.F0 = 0 if F0 is None else F0
        self.rho = material.rho
        self.E = material.E

        self.S = np.pi * (self.R ** 2 - self.r ** 2)        # m**4
        self.I = (np.pi / 4) * (self.R ** 4 - self.r ** 4)  # m**4

    def G(self, forma:FuncaoForma):
        a = 2 * self.rho * self.I * forma.intg_g2()
        return np.array([
            [    0,    a],
            [   -a,    0],
        ])
        
    def K(self, forma:FuncaoForma):
        ke = self.E * self.I * forma.intg_h2()
        k0 = self.F0 * forma.intg_g2()
        return np.array([
            [  ke + k0,          0],
            [        0,    ke + k0],
        ])
    
    def M(self, forma:FuncaoForma):
        m_ii = self.rho * self.S * forma.intg_f2() + self.rho * self.I * forma.intg_g2()
        return np.array([
            [ m_ii,    0],
            [    0, m_ii]
        ])
    
    def C(self, forma:FuncaoForma):
        alpha = 0
        beta = 0
        return self.M(forma) * alpha + self.K(forma) * beta
    
class Disco:
    def __init__(self, R, d, pos, material:Material, r=None):
        self.R = R
        self.d = d
        self.pos = pos
        self.r = 0 if r is None else r

        self.M_d = np.pi * (self.R ** 2 - self.r  ** 2) * d * material.rho                  # kg
        self.I_d_x = (self.M_d / 12) * (3 * self.R ** 2 + 3 * self.r ** 2 + self.d ** 2)    # kg.m²
        self.I_d_z = (self.M_d / 2) * (self.R ** 2 + self.r ** 2)                           # kg.m²

    def M(self, forma:FuncaoForma):
        m_ii = self.M_d * (forma.f(self.pos) ** 2) + self.I_d_x * (forma.g(self.pos) ** 2)
        return np.array([
            [ m_ii,    0],
            [    0, m_ii]
        ])

    def G(self, forma:FuncaoForma):
        a = self.I_d_z * (forma.g(self.pos) ** 2)
        return np.array([
            [    0,    a],
            [   -a,    0],
        ])
    
class ForcaAssincrona:
    def __init__(self, Fx, Fy, pos, s):
        self.Fx = Fx
        self.Fy = Fy
        self.pos = pos
        self.s = s

        if Fx == Fy:
            self.gdl = "ambos"
        else:
            self.gdl = "separado"

    def magnitude(self, forma:FuncaoForma):
        return self.Fx * forma.f(self.pos), self.Fy * forma.f(self.pos)
    
class Desbalanceamento:
    def __init__(self, m_u, d=None, disco:Disco=None, pos=None):
        self.m_u = m_u
        if disco is None:
            if d is None or pos is None:
                raise Exception("Como não inseriu um disco, passe a posição do desbalançeamento e raio.")
            else:
                self.d = d
                self.pos = pos
        else:
            self.d = disco.R
            self.pos = disco.pos

        self.magnitude = self.m_u * self.d

    def F(self, omega:float, forma:FuncaoForma):
        return self.magnitude * (omega ** 2) * forma.f(self.pos)


class Mancal:
    def __init__(self, kxx, kyy, cxx, cyy, pos, kxy=None, kyx=None, cxy=None, cyx=None):
        self.kxx = kxx
        self.kyy = kyy
        self.cxx = cxx
        self.cyy = cyy
        self.pos = pos

        self.kxy = 0 if kxy is None else kxy
        self.kyx = 0 if kyx is None else kyx
        self.cxy = 0 if cxy is None else cxy
        self.cyx = 0 if cyx is None else cyx

    def K(self, forma:FuncaoForma):
        f = forma.f(self.pos) ** 2
        kxx = self.kxx * f
        kyy = self.kyy * f
        kxy = self.kxy * f
        kyx = self.kyx * f

        return np.array([
            [kxx, kxy],
            [kyx, kyy],
        ])

    def C(self, forma:FuncaoForma):
        f = forma.f(self.pos) ** 2
        cxx = self.cxx * f
        cyy = self.cyy * f
        cxy = self.cxy * f
        cyx = self.cyx * f

        return np.array([
            [cxx, cxy],
            [cyx, cyy],
        ])

    
class Rotor:
    def __init__(self, eixo:Eixo, forma:FuncaoForma, mancal=None, disco=None, desbalanceamento=None, gdl=1, forca_assincrona:ForcaAssincrona=None):
        self.eixo = eixo
        self.mancal = mancal
        self.disco = disco
        self.forma = forma
        self.desbalanceamento = desbalanceamento
        self.forca_assincrona = forca_assincrona
        self.n_gdl = int(2 * gdl)

    def M(self,):
        M_d = np.zeros((self.n_gdl,self.n_gdl))
        if self.disco is not None:
            if isinstance(self.disco, Disco):
                M_d = self.disco.M(self.forma)
            elif isinstance(self.disco, list):
                for disco in self.disco:
                    M_d += disco.M(self.forma)
        
        return self.eixo.M(self.forma) + M_d

    def G(self,):
        G_d = np.zeros((self.n_gdl,self.n_gdl))
        if self.disco is not None:
            if isinstance(self.disco, Disco):
                G_d = self.disco.G(self.forma)
            elif isinstance(self.disco, list):
                for disco in self.disco:
                    G_d += disco.G(self.forma)

        return self.eixo.G(self.forma) + G_d
    
    def K(self,):
        K_b = np.zeros((self.n_gdl,self.n_gdl))
        if self.mancal is not None:
            if isinstance(self.mancal, Mancal):
                K_b = self.mancal.K(self.forma)
            elif isinstance(self.mancal, list):
                for mancal in self.mancal:
                    K_b += mancal.K(self.forma)
        
        return self.eixo.K(self.forma) + K_b
    
    def C(self,):
        C_b = np.zeros((self.n_gdl,self.n_gdl))
        if self.mancal is not None:
            if isinstance(self.mancal, Mancal):
                C_b = self.mancal.C(self.forma)
            elif isinstance(self.mancal, list):
                for mancal in self.mancal:
                    C_b += mancal.C(self.forma)
        
        return self.eixo.C(self.forma) + C_b
    
    def fn(self, omega_rpm, through="amplitude", excitacao="livre", gdl="ambos"):
        omega = omega_rpm * (2 * np.pi / 60)
        if through == "equation":
            ## Equação Característica
            # m² u² + (2km + a²omega²) u + k²
            # u = r²

            k = self.K()[0,0]
            m = self.M()[0,0]
            a = np.abs(self.G()[0,1])

            b = (2 * k * m + (a ** 2) * (omega ** 2))
            delta = (b ** 2 - 4 * (m ** 2) * (k ** 2))
            den = 2 * (m ** 2)

            num_1 = -b + np.sqrt(delta)
            num_2 = -b - np.sqrt(delta)

            r_2_1 = num_1 / den
            r_2_2 = num_2 / den

            fn_f = np.sqrt(- r_2_2 / (4 * (np.pi ** 2)))
            fn_b = np.sqrt(- r_2_1 / (4 * (np.pi ** 2)))

            return fn_f, fn_b

        elif through == "amplitude":
            if self.forca_assincrona is not None and excitacao not in ["livre", "desbalanceamento"]:
                s = self.forca_assincrona.s
            else:
                s = 1

            w_list = np.linspace(0, 9000, 5000)

            A1 = []
            A2 = []
            for w_rpm in w_list:
                w = w_rpm * (2 * np.pi) / 60
                amp = self.amplitude_vibracao(w, s, excitacao, gdl, omega)
                A1.append(amp[0][0])
                A2.append(amp[1][0])

            A1 = np.array(A1) / max(A1)
            A2 = np.array(A2) / max(A2)
            
            indices_1, _ = find_peaks(A1, 0.1)
            indices_2, _ = find_peaks(A2, 0.1)

            v1_c = w_list[indices_1].tolist()
            v2_c = w_list[indices_2].tolist()

            if len(v1_c) == 1 and len(v2_c):
                fn_b, fn_f = sorted(v1_c + v2_c)
            else:
                fn_b, fn_f = v1_c

            return fn_f, fn_b
    
    def velocidades_criticas(self, excitacao="livre", gdl="ambos"):
        if self.forca_assincrona is not None and excitacao not in ["livre", "desbalanceamento"]:
            s = self.forca_assincrona.s
        else:
            s = 1

        omega_list = np.linspace(0, 9000, 5000)

        A1 = []
        A2 = []
        for omega_rpm in omega_list:
            omega = omega_rpm * (2 * np.pi) / 60
            amp = self.amplitude_vibracao(omega, s, excitacao, gdl, omega)
            A1.append(amp[0][0])
            A2.append(amp[1][0])

        A1 = np.array(A1) / max(A1)
        A2 = np.array(A2) / max(A2)
        
        indices_1, _ = find_peaks(A1, 0.1)
        indices_2, _ = find_peaks(A2, 0.1)

        v1_c = omega_list[indices_1].tolist()
        v2_c = omega_list[indices_2].tolist()

        if len(v1_c) == 1 and len(v2_c):
            fn_b, fn_f = sorted(v1_c + v2_c)
        else:
            fn_b, fn_f = v1_c

        return fn_f, fn_b
    
    def amplitude_vibracao(self, omega, s=1, excitacao="livre", gdl="ambos", omega_rpm=None):
        k1 = self.K()[0,0]
        k2 = self.K()[1,1]

        m = self.M()[0,0]
        a = np.abs(self.G()[0,1])

        if excitacao == "livre":
            B = np.array([
                [0.0001],
                [0.0001]
            ])
        elif excitacao == "desbalanceamento":
            omega_rpm = omega
            s = 1
            F = self.desbalanceamento.F(omega, self.forma)
            B = np.array([
                [F],
                [F]
            ])
        else:
            if gdl == "ambos":
                F = self.forca_assincrona.magnitude(self.forma)[0]
                B = np.array([
                    [F],
                    [F]
                ])
            else:
                F = self.forca_assincrona.magnitude(self.forma)
                B = np.array([
                    [F[0]],
                    [F[1]]
                ])

        w = omega * s 
        A = np.array([
            [    k1 - m * (w ** 2),   a * w * omega_rpm],
            [a * w * omega_rpm,       k2 - m * (w ** 2)],
        ])

        amp = la.solve(A, B)
    
        return amp
    
    def plot_resposta(self, speed_range:tuple, n_points=1000, tipo="desbalanceamento", w_rpm=None):
        if tipo != "desbalanceamento":
            s = self.forca_assincrona.s
            gdl = self.forca_assincrona.gdl
        else:
            s = 1
            gdl = "ambos"
        
        omega_c_rpm = self.velocidade_critica_forcada(s, gdl, w_rpm)
        fig = go.Figure()
        omega_list = np.linspace(speed_range[0], speed_range[1], num=n_points, endpoint=True)

        amplitudes = {}
        amplitude_dict = {
            "w_1" : {
                "amplitude" : [],
                "omega" : [],
            },
            "w_2" : {
                "amplitude" : [],
                "omega" : [],
            },
        }

        if gdl == "ambos":
            amplitudes["A1"] = amplitude_dict
        else:
            amplitudes["A1"] = amplitude_dict
            amplitudes["A2"] = amplitude_dict

        cores = [
            "#4787FF",
            "#FF4747"
        ]
        for i, omega_rpm in enumerate(omega_list):
            if np.isclose(omega_rpm, omega_c_rpm):
                pass
            else:
                amp = np.abs(
                    self.amplitude_vibracao(
                        omega=omega_rpm * (2 * np.pi) / 60,
                        s=s,
                        excitacao=tipo,
                        gdl=gdl,
                        omega_rpm=w_rpm
                    )
                )
                if omega_rpm < omega_c_rpm:

                    if isinstance(amp, tuple):
                        idx = 0
                        for k, v in amplitudes.items():
                            v["w_1"]["amplitude"].append(amp[idx])
                            v["w_1"]["omega"].append(omega_rpm)
                            idx += 1
                    else:
                        amplitudes["A1"]["w_1"]["amplitude"].append(amp)
                        amplitudes["A1"]["w_1"]["omega"].append(omega_rpm)
                elif omega_rpm > omega_c_rpm:
                    if isinstance(amp, tuple):
                        idx = 0
                        for k, v in amplitudes.items():
                            v["w_2"]["amplitude"].append(amp[idx])
                            v["w_2"]["omega"].append(omega_rpm)
                            idx += 1
                    else:
                        amplitudes["A1"]["w_2"]["amplitude"].append(amp)
                        amplitudes["A1"]["w_2"]["omega"].append(omega_rpm)

        max_amp = np.max(amplitudes["A1"]["w_1"]["amplitude"] + amplitudes["A1"]["w_2"]["amplitude"])
        idx = 0
        for name, amplitude in amplitudes.items():
            cont_trace=0
            for k, v in amplitude.items():
                fig.add_trace(go.Scatter(
                    x=v["omega"],
                    y=v["amplitude"],
                    mode="lines",
                    name=name,
                    line=dict(
                        color=cores[idx],
                        width=2,
                    ),
                    legendgroup=name,
                    showlegend=True if cont_trace == 0 else False,
                ))
                cont_trace += 1
            idx += 1

        fig.add_vline(
            x=omega_c_rpm,
            line_width=2, 
            line_dash="dash",
            line_color="red",
            annotation_text=f"Velocidade Crítica {omega_c_rpm:.3f} RPM"
        )
        fig.update_layout(
            title=f"<b>Resposta a(o) {tipo.replace('_', ' ')}</b>",
        )
        fig.update_xaxes(
            title="Velocidade [rpm]",
        )
        fig.update_yaxes(
            title="Amplitude [m]",
            range=[0, max_amp / 50],
        )
        return fig        

    def plot_campbell(self, speed_range:tuple, n_points=100, through="equation", title=None):
        fig = go.Figure()
        omega_list = np.linspace(speed_range[0], speed_range[1], num=n_points, endpoint=True)
        fn_f = np.zeros(len(omega_list))
        fn_b = np.zeros(len(omega_list))
        curve_Nx = {
            "1x": np.zeros(len(omega_list)),
        }
        if self.forca_assincrona is not None:
            curve_Nx[f"{self.forca_assincrona.s:.1f}x"]=np.zeros(len(omega_list))


        for i, omega_rpm in enumerate(omega_list):
            fn = self.fn(omega_rpm, through)
            fn_f[i] = fn[0]
            fn_b[i] = fn[1]
            for k, v in curve_Nx.items():
                s = float(k.split("x")[0])
                v[i] = s * omega_rpm / 60

        fig.add_trace(go.Scatter(
            x=omega_list,
            y=fn_f,
            mode="lines",
            line=dict(
                color="#4787FF",
                width=2,
            ),
            name="Forward",
        ))
        fig.add_trace(go.Scatter(
            x=omega_list,
            y=fn_b,
            mode="lines",
            line=dict(
                color="#5574AD",
                width=2,
            ),
            name="Backward",
        ))
        for k, v in curve_Nx.items():
            fig.add_trace(go.Scatter(
                x=omega_list,
                y=v,
                mode="lines",
                line=dict(
                    # color="#FF441F",
                    width=2,
                ),
                name=k,
            ))
        try:
            v_crit_f, v_crit_b = self.velocidades_criticas()
            fig.add_trace(go.Scatter(
                x=[v_crit_f * (60 / (2 * np.pi))],
                y=[v_crit_f / (2 * np.pi)],
                name="Velocidade Crítica - FW",
                mode="markers",
                marker=dict(
                    color="#000000"
                )
            ))
            fig.add_trace(go.Scatter(
                x=[v_crit_b* (60 / (2 * np.pi))],
                y=[v_crit_b / (2 * np.pi)],
                name="Velocidade Crítica - BW",
                mode="markers",
                marker=dict(
                    color="#000000"
                )
            ))
        except (RuntimeError
                ):
            pass

        if title is None:
            fig.update_layout(
                title="Diagrama de Campbell",
            )
        else:
            fig.update_layout(
                title=title,
            )

        fig.update_xaxes(
            title="Velocidade [rpm]"
        )
        fig.update_yaxes(
            title="Frequência [hz]"
        )
        
        return fig