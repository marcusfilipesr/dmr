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
        return -(self.c0**2) * np.sin(self._x(pos))

    def _x(self, z):
        return self.c0 * z

    def intg_f2(
        self,
    ):
        # x / 2 - sin(2 c0 x) / (4 c0)
        up = self.L / 2 - (np.sin(2 * self._x(self.L)) / (4 * self.c0))
        lo = -(np.sin(2 * self._x(0)) / (4 * self.c0))
        return up - lo

    def intg_g2(
        self,
    ):
        # x / 2 + sin(2 c0 x) / (4 c0)
        up = self.L / 2 + (np.sin(2 * self._x(self.L)) / (4 * self.c0))
        lo = np.sin(2 * self._x(0)) / (4 * self.c0)
        return (self.c0**2) * (up - lo)

    def intg_h2(
        self,
    ):
        # x / 2 - sin(2 c0 x) / (4 c0)
        up = self.L / 2 - (np.sin(2 * self._x(self.L)) / (4 * self.c0))
        lo = -(np.sin(2 * self._x(0)) / (4 * self.c0))
        return (self.c0**4) * (up - lo)

    def plot(self, n_points=20):
        fig = go.Figure()
        length = np.linspace(0, self.L, num=n_points)
        forma = np.zeros(len(length))
        for i, z in enumerate(length):
            forma[i] = self.f(z)

        fig.add_trace(
            go.Scatter(
                x=length,
                y=forma,
                name="Função de Forma",
                mode="lines",
                line=dict(color="#000000"),
            )
        )

        return fig


class Eixo:
    def __init__(self, R, L, material: Material, r=None, F0=None):
        self.R = R
        self.L = L
        self.r = 0 if r is None else r
        self.F0 = 0 if F0 is None else F0
        self.rho = material.rho
        self.E = material.E

        self.S = np.pi * (self.R**2 - self.r**2)  # m**4
        self.I = (np.pi / 4) * (self.R**4 - self.r**4)  # m**4

    def G(self, forma: FuncaoForma):
        a = 2 * self.rho * self.I * forma.intg_g2()
        return np.array(
            [
                [0, a],
                [-a, 0],
            ]
        )

    def K(self, forma: FuncaoForma):
        ke = self.E * self.I * forma.intg_h2()
        k0 = self.F0 * forma.intg_g2()
        return np.array(
            [
                [ke + k0, 0],
                [0, ke + k0],
            ]
        )

    def M(self, forma: FuncaoForma):
        m_ii = self.rho * self.S * forma.intg_f2() + self.rho * self.I * forma.intg_g2()
        return np.array([[m_ii, 0], [0, m_ii]])

    def C(self, forma: FuncaoForma):
        alpha = 0
        beta = 0
        return self.M(forma) * alpha + self.K(forma) * beta


class Disco:
    def __init__(self, R, d, pos, material: Material, r=None):
        self.R = R
        self.d = d
        self.pos = pos
        self.r = 0 if r is None else r

        self.M_d = np.pi * (self.R**2 - self.r**2) * d * material.rho  # kg
        self.I_d_x = (self.M_d / 12) * (
            3 * self.R**2 + 3 * self.r**2 + self.d**2
        )  # kg.m²
        self.I_d_z = (self.M_d / 2) * (self.R**2 + self.r**2)  # kg.m²

    def M(self, forma: FuncaoForma):
        m_ii = self.M_d * (forma.f(self.pos) ** 2) + self.I_d_x * (
            forma.g(self.pos) ** 2
        )
        return np.array([[m_ii, 0], [0, m_ii]])

    def G(self, forma: FuncaoForma):
        a = self.I_d_z * (forma.g(self.pos) ** 2)
        return np.array(
            [
                [0, a],
                [-a, 0],
            ]
        )


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

    def magnitude(self, forma: FuncaoForma):
        return self.Fx * forma.f(self.pos), self.Fy * forma.f(self.pos)


class Desbalanceamento:
    def __init__(self, m_u, d=None, disco: Disco = None, pos=None):
        self.m_u = m_u
        if disco is None:
            if d is None or pos is None:
                raise Exception(
                    "Como não inseriu um disco, passe a posição do desbalançeamento e raio."
                )
            else:
                self.d = d
                self.pos = pos
        else:
            self.d = disco.R
            self.pos = disco.pos

        self.magnitude = self.m_u * self.d

    def F(self, omega: float, forma: FuncaoForma):
        return self.magnitude * (omega**2) * forma.f(self.pos)


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

    def K(self, forma: FuncaoForma):
        f = forma.f(self.pos) ** 2
        kxx = self.kxx * f
        kyy = self.kyy * f
        kxy = self.kxy * f
        kyx = self.kyx * f

        return np.array(
            [
                [kxx, kxy],
                [kyx, kyy],
            ]
        )

    def C(self, forma: FuncaoForma):
        f = forma.f(self.pos) ** 2
        cxx = self.cxx * f
        cyy = self.cyy * f
        cxy = self.cxy * f
        cyx = self.cyx * f

        return np.array(
            [
                [cxx, cxy],
                [cyx, cyy],
            ]
        )


class Rotor:
    def __init__(
        self,
        eixo: Eixo,
        forma: FuncaoForma,
        mancal=None,
        disco=None,
        desbalanceamento=None,
        gdl=1,
        forca_assincrona: ForcaAssincrona = None,
    ):
        self.eixo = eixo
        self.mancal = mancal
        self.disco = disco
        self.forma = forma
        self.desbalanceamento = desbalanceamento
        self.forca_assincrona = forca_assincrona
        self.n_gdl = int(2 * gdl)

    def M(
        self,
    ):
        M_d = np.zeros((self.n_gdl, self.n_gdl))
        if self.disco is not None:
            if isinstance(self.disco, Disco):
                M_d = self.disco.M(self.forma)
            elif isinstance(self.disco, list):
                for disco in self.disco:
                    M_d += disco.M(self.forma)

        return self.eixo.M(self.forma) + M_d

    def G(
        self,
    ):
        G_d = np.zeros((self.n_gdl, self.n_gdl))
        if self.disco is not None:
            if isinstance(self.disco, Disco):
                G_d = self.disco.G(self.forma)
            elif isinstance(self.disco, list):
                for disco in self.disco:
                    G_d += disco.G(self.forma)

        return self.eixo.G(self.forma) + G_d

    def K(
        self,
    ):
        K_b = np.zeros((self.n_gdl, self.n_gdl))
        if self.mancal is not None:
            if isinstance(self.mancal, Mancal):
                K_b = self.mancal.K(self.forma)
            elif isinstance(self.mancal, list):
                for mancal in self.mancal:
                    K_b += mancal.K(self.forma)

        return self.eixo.K(self.forma) + K_b

    def C(
        self,
    ):
        C_b = np.zeros((self.n_gdl, self.n_gdl))
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

            k = self.K()[0, 0]
            m = self.M()[0, 0]
            a = np.abs(self.G()[0, 1])

            b = 2 * k * m + (a**2) * (omega**2)
            delta = b**2 - 4 * (m**2) * (k**2)
            den = 2 * (m**2)

            num_1 = -b + np.sqrt(delta)
            num_2 = -b - np.sqrt(delta)

            r_2_1 = num_1 / den
            r_2_2 = num_2 / den

            fn_f = np.sqrt(-r_2_2 / (4 * (np.pi**2))) * 60
            fn_b = np.sqrt(-r_2_1 / (4 * (np.pi**2))) * 60

            return fn_f, fn_b

        elif through == "amplitude":
            if self.forca_assincrona is not None and excitacao not in [
                "livre",
                "desbalanceamento",
            ]:
                s = self.forca_assincrona.s
            else:
                s = 1

            w_list = np.linspace(0, 9000, 5000)

            A1 = []
            A2 = []
            for w_rpm in w_list:
                w = w_rpm * (2 * np.pi) / 60
                amp = self.amplitude_vibracao(w, s, excitacao, gdl, omega)
                if amp[0] == 0 and amp[1] != 0:
                    A1.append(amp[1])
                    A2.append(amp[1])
                elif amp[1] == 0 and amp[0] != 0:
                    A1.append(amp[0])
                    A2.append(amp[0])
                else:
                    A1.append(amp[0])
                    A2.append(amp[1])

            if max(A1) != 0:
                A1 = np.array(A1) / max(A1)
                indices_1, _ = find_peaks(A1, 0.1)
            else:
                A1 = np.array(A1)
                indices_1, _ = find_peaks(A1)

            if max(A2) != 0:
                A2 = np.array(A2) / max(A2)
                indices_2, _ = find_peaks(A2, 0.1)
            else:
                A2 = np.array(A2)
                indices_2, _ = find_peaks(A2)

            v1_c = w_list[indices_1].tolist()
            v2_c = w_list[indices_2].tolist()

            if len(v1_c) == 1 and len(v2_c) == 1:
                fn_b, fn_f = sorted(v1_c + v2_c)
            elif len(v1_c) > 1:
                fn_b, fn_f = v1_c
            else:
                fn_b, fn_f = v2_c

            return fn_f, fn_b

    def velocidades_criticas(self, excitacao="livre", gdl="ambos", through="amplitude"):
        if through == "amplitude":
            if self.forca_assincrona is not None and excitacao not in [
                "livre",
                "desbalanceamento",
            ]:
                s = self.forca_assincrona.s
            else:
                s = 1

            omega_list = np.linspace(0, 9000, 5000)

            A1 = []
            A2 = []
            for omega_rpm in omega_list:
                omega = omega_rpm * (2 * np.pi) / 60
                amp = self.amplitude_vibracao(omega, s, excitacao, gdl, omega)
                A1.append(amp[0])
                A2.append(amp[1])

            A1 = np.array(A1) / max(A1)
            A2 = np.array(A2) / max(A2)

            indices_1, _ = find_peaks(A1, 0.1)
            indices_2, _ = find_peaks(A2, 0.1)

            v1_c = omega_list[indices_1].tolist()
            v2_c = omega_list[indices_2].tolist()

            if len(v1_c) == 1 and len(v2_c) == 1:
                fn_b, fn_f = sorted(v1_c + v2_c)
            elif len(v1_c) > 1:
                fn_b, fn_f = v1_c
            else:
                fn_b, fn_f = v2_c

        else:
            k = self.K()[0, 0]
            m = self.M()[0, 0]
            a = np.abs(self.G()[0, 1])

            fn_b = np.sqrt(k / (m + a)) * 60 / (2 * np.pi)
            fn_f = np.sqrt(k / (m - a)) * 60 / (2 * np.pi)

        return fn_f, fn_b

    def amplitude_vibracao(
        self, omega, s=1, excitacao="livre", gdl="ambos", omega_rpm=None
    ):
        k1 = self.K()[0, 0]
        k2 = self.K()[1, 1]
        m = self.M()[0, 0]
        a = np.abs(self.G()[0, 1])

        if excitacao == "livre":
            B = np.array([[0.0001], [0.0001]])
        elif excitacao == "desbalanceamento":
            s = 1
            F = self.desbalanceamento.F(omega, self.forma)
            B = np.array([F, F])
        else:
            if gdl == "ambos":
                F = self.forca_assincrona.magnitude(self.forma)[0]
                B = np.array([F, F])
            else:
                F = self.forca_assincrona.magnitude(self.forma)
                B = np.array([F[0], F[1]])

        w = omega * s
        A = np.array(
            [
                [  k1 - m * (w**2), a * w * omega_rpm],
                [a * w * omega_rpm,   k2 - m * (w**2)],
            ]
        )

        amp = la.lstsq(A, B, cond=1e-6)[0]

        return amp.flatten()

    def F(self, omega, t, s=1, excitacao="livre", gdl="ambos"):
        if excitacao == "livre":
            B = np.array([[0.0001], [0.0001]])
        elif excitacao == "desbalanceamento":
            s = 1
            F = self.desbalanceamento.F(omega, self.forma)
            B = np.array([F , F])
        else:
            if gdl == "ambos":
                F = self.forca_assincrona.magnitude(self.forma)[0]
                B = np.array([F, F])
            else:
                F = self.forca_assincrona.magnitude(self.forma)
                B = np.array([F[0], F[1]])

        aux = s * omega * t
        return np.array([B[0] * np.cos(aux), B[1] * np.sin(aux)])

    def inverse_mass_matrix(self,):
        self.M_inv = np.linalg.inv(self.M())

    def A(self, omega):
        M_inv = self.M_inv
        A = np.zeros((2 * self.n_gdl, 2 * self.n_gdl))
        A[0:self.n_gdl, self.n_gdl:2*self.n_gdl] = np.eye(self.n_gdl)
        A[self.n_gdl:2*self.n_gdl, 0:self.n_gdl] = - M_inv @ self.K()
        A[self.n_gdl:2*self.n_gdl, self.n_gdl:2*self.n_gdl] = - M_inv @ (self.C() + omega * self.G())

        return A

    def B(self, t, M_inv, omega, s, excitacao, gdl):
        B = np.zeros(2 * self.n_gdl)
        F = self.F(omega, t, s, excitacao, gdl)
        B[self.n_gdl:2*self.n_gdl] = (M_inv @ F).flatten()
        return B
    
    def dydt(self, t, y, A, M_inv, omega, s, excitacao, gdl):
        B = self.B(t, M_inv, omega, s, excitacao, gdl)
        return A @ y + B

    def RK4(self, t, y, h, A, M_inv, omega, s, excitacao, gdl):
        k1 = self.dydt(t,         y,          A, M_inv, omega, s, excitacao, gdl)
        k2 = self.dydt(t + h/2,   y + h/2*k1, A, M_inv, omega, s, excitacao, gdl)
        k3 = self.dydt(t + h/2,   y + h/2*k2, A, M_inv, omega, s, excitacao, gdl)
        k4 = self.dydt(t + h,     y + h*k3,   A, M_inv, omega, s, excitacao, gdl)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    def FRF(self, omega_range=(0,9000), n_points=1000):
        I = np.eye(2 * self.n_gdl)
        omega_list = np.linspace(omega_range[0], omega_range[1], n_points)
        FRF = np.array((2 * self.n_gdl, n_points))
        for i, omega in enumerate(omega_list):
            FRF[:, i] = np.diag(np.linalg.inv(1j * omega * I - self.A(omega))) # @ self.B(t, self.M_inv, omega, s, excitacao, gdl)
        return FRF

    def resposta_temporal(self, omega, t_inicio=0.0, t_fim=1.0, dt=1e-4, excitacao="livre", gdl="ambos", omega_rpm=None):
        if self.forca_assincrona is not None and excitacao not in [
            "livre",
            "desbalanceamento",
        ]:
            s = self.forca_assincrona.s
            gdl = self.forca_assincrona.gdl
        else:
            s = 1
            gdl = "ambos"

        t = np.arange(t_inicio, t_fim, dt)
        n = len(t)
        Y = np.zeros((2 * self.n_gdl, n))
        Y[:, 0] = np.array([0.0, 0.0, 0.0, 0.0])

        A = self.A(omega)
        M_inv = self.M_inv
        for i in range(1, n):
            if excitacao == "assincrona" and gdl == "separado":
                w = omega_rpm * (2 * np.pi / 60)
            else:
                w = omega
            Y[:, i] = self.RK4(
                t=t[i-1],
                y=Y[:, i-1],
                h=dt,
                A=A,
                M_inv=M_inv,
                omega=w,
                s=s,
                excitacao=excitacao,
                gdl=gdl,
            )

        return Y[0, :], Y[1, :]

    def amplitude_sinal_temporal(self, omega, t_fim=1, dt=1e-4, excitacao="livre", gdl="ambos", omega_rpm=None):
                
        q1, q2 = self.resposta_temporal(
            omega=omega,
            t_inicio=0,
            t_fim=t_fim,
            dt=dt,
            excitacao=excitacao,
            gdl=gdl,
            omega_rpm=omega_rpm
        )

        n_pontos = t_fim / dt

        idx_permanente = int(0.9 * n_pontos)
        q1_ss = q1[idx_permanente:]
        q2_ss = q2[idx_permanente:]

        amp_q1 = (q1_ss.max() - q1_ss.min()) / 2
        amp_q2 = (q2_ss.max() - q2_ss.min()) / 2
        # amp_orbital = np.sqrt(q1_ss**2 + q2_ss**2).max()

        return amp_q1, amp_q2

    def plot_resposta(
        self,
        speed_range: tuple,
        n_points=5000,
        excitacao="desbalanceamento",
        w_rpm=None,
        speed_unit="rpm",
    ):
        if excitacao != "desbalanceamento":
            s = self.forca_assincrona.s
            gdl = self.forca_assincrona.gdl
        else:
            s = 1
            gdl = "ambos"

        if excitacao == "assincrona" and gdl == "separado":
            omega_c_rpm = self.fn(
                omega_rpm=w_rpm,
                excitacao=excitacao,
                gdl=gdl,
            )
        else:
            omega_c_rpm = self.velocidades_criticas(
                excitacao=excitacao,
                gdl=gdl,
            )
            
        fig = go.Figure()
        omega_list = np.linspace(
            speed_range[0], speed_range[1], num=n_points, endpoint=True
        )

        amplitudes = {}
        amplitude_dict = {
            "amplitude": [],
            "omega": [],
        }

        amplitudes["A1"] = amplitude_dict
        amplitudes["A2"] = amplitude_dict

        cores = ["#4787FF", "#FF4747"]
        for i, omega_rpm in enumerate(omega_list):
            if excitacao == "assincrona" and gdl == "separado":
                pass
            else:
                w_rpm = omega_rpm

            amp = np.abs(
                self.amplitude_vibracao(
                    omega=omega_rpm * (2 * np.pi) / 60,
                    s=s,
                    excitacao=excitacao,
                    gdl=gdl,
                    omega_rpm=w_rpm * (2 * np.pi) / 60,
                )
            )
            amplitudes["A1"]["amplitude"].append(amp[0])
            amplitudes["A2"]["amplitude"].append(amp[1])

            amplitudes["A1"]["omega"].append(omega_rpm if speed_unit == "rpm" else omega_rpm / 60)
            amplitudes["A2"]["omega"].append(omega_rpm if speed_unit == "rpm" else omega_rpm / 60)

        max_amp = np.max(amplitudes["A1"]["amplitude"])
        idx = 0
        for name, amplitude in amplitudes.items():
            fig.add_trace(
                go.Scatter(
                    x=amplitude["omega"],
                    y=amplitude["amplitude"],
                    mode="lines",
                    name=name,
                    line=dict(
                        color=cores[idx],
                        width=2,
                    ),
                    legendgroup=name,
                    showlegend=True,
                )
            )
            idx += 1

        for w_c in omega_c_rpm:
            fig.add_vline(
                x=w_c if speed_unit == "rpm" else w_c / 60,
                line_width=2,
                line_dash="dash",
                line_color="black",
                annotation_text=f"{w_c if speed_unit == 'rpm' else w_c / 60:.1f} {speed_unit}",
            )

        fig.update_layout(
            title=f"<b>Resposta a(o) {excitacao.replace('_', ' ')}</b>",
        )
        fig.update_xaxes(
            title=f"Velocidade [{speed_unit}]",
        )
        fig.update_yaxes(
            title="Amplitude [m]",
            range=[0, max_amp / 80],
        )
        return fig

    def plot_campbell(
        self, speed_range: tuple, n_points=100, through="amplitude", title=None
    ):
        fig = go.Figure()
        omega_list = np.linspace(
            speed_range[0], speed_range[1], num=n_points, endpoint=True
        )
        fn_f = np.zeros(len(omega_list))
        fn_b = np.zeros(len(omega_list))
        curve_Nx = {
            "1x": np.zeros(len(omega_list)),
        }
        if self.forca_assincrona is not None and self.forca_assincrona.s != 1:
            curve_Nx[f"{self.forca_assincrona.s:.1f}x"] = np.zeros(len(omega_list))

        for i, omega_rpm in enumerate(omega_list):
            fn = self.fn(omega_rpm, through, excitacao="assincrona", gdl="separado")
            fn_f[i] = fn[0] / 60
            fn_b[i] = fn[1] / 60
            for k, v in curve_Nx.items():
                s = float(k.split("x")[0])
                v[i] = s * omega_rpm / 60

        fig.add_trace(
            go.Scatter(
                x=omega_list,
                y=fn_f,
                mode="lines",
                line=dict(
                    color="#4787FF",
                    width=2,
                ),
                name="Forward",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=omega_list,
                y=fn_b,
                mode="lines",
                line=dict(
                    color="#5574AD",
                    width=2,
                ),
                name="Backward",
            )
        )
        for k, v in curve_Nx.items():
            fig.add_trace(
                go.Scatter(
                    x=omega_list,
                    y=v,
                    mode="lines",
                    line=dict(
                        # color="#FF441F",
                        width=2,
                    ),
                    name=k,
                )
            )
        v_crit_f, v_crit_b = self.velocidades_criticas(
            excitacao="assincrona",
            gdl="separado",
            through=through,
        )
        fig.add_trace(
            go.Scatter(
                x=[v_crit_f],
                y=[v_crit_f / 60],
                name="Velocidade Crítica - FW",
                mode="markers",
                marker=dict(color="#000000"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[v_crit_b],
                y=[v_crit_b / 60],
                name="Velocidade Crítica - BW",
                mode="markers",
                marker=dict(color="#000000"),
            )
        )

        if title is None:
            fig.update_layout(
                title="Diagrama de Campbell",
            )
        else:
            fig.update_layout(
                title=title,
            )

        fig.update_xaxes(title="Velocidade [rpm]")
        fig.update_yaxes(title="Frequência [hz]")
        return fig

    def plot_orbita(
            self,
            omega_list:list,
            excitacao="desbalanceamento",
            gdl="ambos",
            omega_rpm=None,
        ):
        if excitacao != "desbalanceamento":
            s = self.forca_assincrona.s
            gdl = self.forca_assincrona.gdl
        else:
            s = 1
            gdl = "ambos"

        fig_dict = {}
        for omega in omega_list:
            omega *= (2 * np.pi) / 60
            if excitacao == "assincrona" and gdl == "separado":
                pass
            else:
                omega_rpm = omega
            A1, A2 = self.amplitude_vibracao(
                omega=omega * (2 * np.pi) / 60,
                s=s,
                excitacao=excitacao,
                gdl=gdl,
                omega_rpm=omega_rpm * (2 * np.pi) / 60,
            )

            if np.sign(A1 * A2) == -1:
                precessao = "BW"
            else:
                if np.sign(A1) == 1:
                    precessao = "FW"
                else:
                    precessao = "BW"

            T = (2 * np.pi) / omega
            dt = T / 10
            t = np.linspace(0, T - dt, 100)
            q1 = A1 * np.cos(s * omega * t)
            q2 = A2 * np.sin(s * omega * t)

            max_axis = np.max([np.max(q1), np.max(q2)])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=q1,
                    y=q2,
                    name="Orbita",
                    mode="lines",
                    line={
                        "color": "#4787FF",
                        "width": 2,
                    },
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[q1[-1]],
                    y=[q2[-1]],
                    name=precessao,
                    mode="markers",
                    line={
                        "color": "#4787FF",
                    },
                )
            )
            fig.update_layout(
                title=f"<b>Órbita</b> {omega * 60 / (2 * np.pi):.0f} RPM",
                autosize=False,
                width=400,
                height=400,
            )
            fig.update_xaxes(
                title=r'$\text{Deslocamento } q_{1}$',
                range=[-max_axis, max_axis]
            )
            fig.update_yaxes(
                title=r'$\text{Deslocamento } q_{2}$',
                range=[-max_axis, max_axis]
            )

            fig_dict[f"{omega * 60 / (2 * np.pi):.0f}RPM"] = fig

        return fig_dict