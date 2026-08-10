import numpy as np
import plotly.graph_objects as go


from plotly.subplots import make_subplots
from scipy.optimize import newton, root
from scipy.signal import find_peaks
from scipy import linalg as la

tableau_colors = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

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
                [ 0, a],
                [-a, 0],
            ]
        )

    def Kst(self, forma: FuncaoForma):
        a = 2 * self.rho * self.I * forma.intg_g2()
        return np.array(
            [
                [0, a],
                [0, 0],
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
                [ 0, a],
                [-a, 0],
            ]
        )

    def Kst(self, forma: FuncaoForma):
        a = self.I_d_z * (forma.g(self.pos) ** 2)
        return np.array(
            [
                [0, a],
                [0, 0],
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

    def F(self, forma: FuncaoForma):
        return self.magnitude * forma.f(self.pos)


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

class Omega:
    def __init__(self, omega_0, omega_f, t_0, t_f, t_sim=None, tipo="linear", lbd=None):
        self.tipo = tipo
        if tipo == "linear":
            self.A = (omega_0 * t_f - omega_f * t_0) / (t_f - t_0)
            self.B = (omega_0 - omega_f) / (t_0 - t_f)
        elif tipo == "exponencial":
            if lbd is None:
                lbd = 0.5
            self.A = (omega_f * np.exp(lbd * t_f) - omega_0 * np.exp(lbd * t_0)) / (np.exp(lbd * t_f) - np.exp(lbd * t_0))
            self.B = (omega_f - omega_0) / (np.exp(-lbd * t_f) - np.exp(-lbd * t_0))
        self.lbd = lbd
        self.t_f = t_f
        self.t_0 = t_0
        if t_sim is None:
            t_sim = t_f
        self.t_sim = t_sim

    def t(self, t):
        return min(t, self.t_f)
        
    def v(self, t):
        if self.tipo == "linear":
            return self.A + self.B * self.t(t)
        elif self.tipo == "exponencial":
            return self.A + self.B * np.exp(- self.lbd * self.t(t))

    def dot(self, t):
        if t > self.t_f:
            return 0
        if self.tipo == "linear":
                return self.B
        elif self.tipo == "exponencial":
            return - self.lbd * self.B * np.exp(- self.lbd * self.t(t))

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

    def Kst(self,):
        Kst_d = np.zeros((self.n_gdl, self.n_gdl))
        if self.disco is not None:
            if isinstance(self.disco, Disco):
                Kst_d = self.disco.Kst(self.forma)
            elif isinstance(self.disco, list):
                for disco in self.disco:
                    Kst_d += disco.Kst(self.forma)

        return self.eixo.Kst(self.forma) + Kst_d

    def fn(self, omega_rpm, through="estados", excitacao="livre", gdl="ambos"):
        omega = omega_rpm * (2 * np.pi / 60)
        if self.forca_assincrona is not None and excitacao not in [
            "livre",
            "desbalanceamento",
        ]:
            s = self.forca_assincrona.s
        else:
            s = 1
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

        elif through == "amplitude":

            w_list = np.linspace(0, 9000, 5000)

            A1 = []
            A2 = []
            for w_rpm in w_list:
                w = w_rpm * (2 * np.pi) / 60
                amp = self.amplitude_vibracao(w, s, excitacao, gdl, omega) 
                #         0   1   2   3 
                # amp = [A1, B1, A2, B2]
                amp1 = np.sqrt(amp[0] ** 2 + amp[1] ** 2)
                amp2 = np.sqrt(amp[2] ** 2 + amp[3] ** 2)

                if amp1 == 0 and amp2 != 0:
                    A1.append(np.abs(amp2))
                    A2.append(np.abs(amp2))
                elif amp2 == 0 and amp1 != 0:
                    A1.append(np.abs(amp1))
                    A2.append(np.abs(amp1))
                else:
                    A1.append(np.abs(amp1))
                    A2.append(np.abs(amp2))

            if max(A1) != 0:
                A1 = (np.array(A1) - np.min(A1)) / (np.max(A1) - np.min(A1))
                indices_1, _ = find_peaks(A1, 0.1)
            else:
                A1 = np.array(A1)
                indices_1, _ = find_peaks(A1)

            if max(A2) != 0:
                A2 = (np.array(A2) - np.min(A2)) / (np.max(A2) - np.min(A2))
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
            elif len(v2_c) > 1:
                fn_b, fn_f = v2_c
            elif not v1_c and len(v2_c) == 1:
                fn_b = v2_c[0]
                fn_f = v2_c[0]
            elif not v2_c and len(v1_c) == 1:
                fn_b = v1_c[0]
                fn_f = v1_c[0]

        elif through == "estados":
            w = s * omega_rpm * (2 * np.pi) / 60
            av = la.eig(self.A(w))[0]
            wn = np.imag(av)
            fn_f = wn[0] * 60 / (2 * np.pi)
            fn_b = wn[2] * 60 / (2 * np.pi)

        return fn_f, fn_b

    def velocidades_criticas(self, excitacao="livre", gdl="ambos", through="estados"):
        if self.forca_assincrona is not None and excitacao not in [
            "livre",
            "desbalanceamento",
        ]:
            s = self.forca_assincrona.s
        else:
            s = 1

        fn_f, fn_b = 0, 0

        if through == "amplitude":

            omega_list = np.linspace(0, 9000, 5000)

            A1 = []
            A2 = []
            for omega_rpm in omega_list:
                omega = omega_rpm * (2 * np.pi) / 60
                amp = self.amplitude_vibracao(omega, s, excitacao, gdl, omega)
                #         0   1   2   3 
                # amp = [A1, B1, A2, B2]
                A1.append(np.sqrt(amp[0] ** 2 + amp[1] ** 2))
                A2.append(np.sqrt(amp[2] ** 2 + amp[3] ** 2))

            A1 = (np.array(A1) - np.min(A1)) / (np.max(A1) - np.min(A1))
            A2 = (np.array(A2) - np.min(A2)) / (np.max(A2) - np.min(A2))

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

        elif through == "equation":
            k = self.K()[0, 0]
            m = self.M()[0, 0]
            a = np.abs(self.G()[0, 1])

            fn_b = np.sqrt(k / (m + a)) * 60 / (2 * np.pi)
            fn_f = np.sqrt(k / (m - a)) * 60 / (2 * np.pi)

        elif through == "estados":
            omega_list = np.linspace(0, 9000, 5000)
            for omega_rpm in omega_list:
                omega = omega_rpm * (2 * np.pi) / 60
                av = la.eig(self.A(omega))[0]
                fn = np.imag(av) * 60 / (2 * np.pi)
                if np.isclose(fn[0], omega_rpm * s, atol=10):  # wn = s * fn
                    fn_f = fn[0] / s
                if np.isclose(fn[2], omega_rpm * s, atol=10):
                    fn_b = fn[2] / s

        return fn_f, fn_b

    def amplitude_vibracao(
        self, omega, s=1, excitacao="livre", gdl="ambos", omega_rpm=None, omega_dot=0
    ):
        k1 = self.K()[0, 0]
        k2 = self.K()[1, 1]
        c1 = self.C()[0, 0]
        c2 = self.C()[1, 1]
        m = self.M()[0, 0]
        a = np.abs(self.G()[0, 1])

        if excitacao == "livre":
            B = np.array([0.0001, 0.0001, 0.0001, 0.0001])
        elif excitacao == "desbalanceamento":
            s = 1
            F = self.desbalanceamento.F(self.forma)
            B = np.array([F * (omega ** 2), (F * omega_dot), - (F * omega_dot), F * (omega ** 2)])
        else:
            if gdl == "ambos":
                F = self.forca_assincrona.magnitude(self.forma)[0]
                B = np.array([F, 0, 0, F])
            else:
                F = self.forca_assincrona.magnitude(self.forma)
                B = np.array([F[0], 0, 0, F[1]])

        w = omega * s
        A = np.array(
            [
                [  k1 - m * (w**2),                c1 * w,        a * omega_dot,    a * w * omega_rpm],
                [         - c1 * w,       k1 - m * (w**2),  - a * w * omega_rpm,        a * omega_dot],
                [                0,   - a * w * omega_rpm,      k2 - m * (w**2),               c2 * w],
                [a * w * omega_rpm,                     0,             - c2 * w,      k2 - m * (w**2)],
            ]
        )

        amp = la.lstsq(A, B, cond=1e-6)[0]

        return amp.flatten()

    def F(self, omega, t, s=1, excitacao="livre", gdl="ambos", omega_dot=0):
        if excitacao == "livre":
            B = np.array([[0.0001], [0.0001]])
        elif excitacao == "desbalanceamento":
            s = 1
            F = self.desbalanceamento.F(self.forma)
            aux = s * omega * t
            return np.array([
                F * ((omega ** 2) * np.cos(aux) + omega_dot * np.sin(aux)),
                F * ((omega ** 2) * np.sin(aux) - omega_dot * np.cos(aux))
            ])
        else:
            if gdl == "ambos":
                F = self.forca_assincrona.magnitude(self.forma)[0]
                B = np.array([F, F])
            else:
                F = self.forca_assincrona.magnitude(self.forma)
                B = np.array([F[0], F[1]])

        aux = s * omega * t
        return np.array([B[0] * np.cos(aux), B[1] * np.sin(aux)])

    def A(self, omega, omega_dot=0):
        Z = np.zeros((self.n_gdl, self.n_gdl))
        I = np.eye(self.n_gdl)

        A = np.vstack(
            [
                np.hstack([Z, I]),
                np.hstack([la.solve(-self.M(), (self.K() + omega_dot * self.Kst())), la.solve(-self.M(), (self.C() + omega * self.G()))])
            ]
        )

        return A

    def B(self, t, omega, s, excitacao, gdl, omega_dot):
        F = self.F(omega, t, s, excitacao, gdl, omega_dot)
        Z = np.zeros(self.n_gdl)
        B = np.concat([Z, la.solve(self.M(), F).flatten()])
        return B
    
    def dydt(self, t, y, A, omega, s, excitacao, gdl, omega_dot):
        B = self.B(t, omega, s, excitacao, gdl, omega_dot)
        return A @ y + B

    def RK4(self, t, y, h, A, omega, s, excitacao, gdl, omega_dot):
        k1 = self.dydt(t,         y,          A, omega, s, excitacao, gdl, omega_dot)
        k2 = self.dydt(t + h/2,   y + h/2*k1, A, omega, s, excitacao, gdl, omega_dot)
        k3 = self.dydt(t + h/2,   y + h/2*k2, A, omega, s, excitacao, gdl, omega_dot)
        k4 = self.dydt(t + h,     y + h*k3,   A, omega, s, excitacao, gdl, omega_dot)
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
        if isinstance(omega, Omega):
            t = np.arange(omega.t_0, omega.t_sim + dt, dt)
        else:
            t = np.arange(t_inicio, t_fim + dt, dt)
        
        n = len(t)
        Y = np.zeros((2 * self.n_gdl, n))
        Y[:, 0] = np.array([0.0, 0.0, 0.0, 0.0])

        if not isinstance(omega, Omega):
            A = self.A(omega * (2 * np.pi / 60))

        omega_dot = 0
        for i in range(1, n):
            if excitacao == "assincrona" and gdl == "separado":
                w = omega_rpm * (2 * np.pi / 60)
            else:
                if isinstance(omega, Omega):
                    w = omega.v(t[i-1]) * (2 * np.pi / 60)
                else:
                    w = omega * (2 * np.pi / 60)

            if isinstance(omega, Omega):
                omega_dot = omega.dot(t[i-1]) * (2 * np.pi / 60)
                A = self.A(w, omega_dot)
            
            Y[:, i] = self.RK4(
                t=t[i-1],
                y=Y[:, i-1],
                h=dt,
                A=A,
                omega=w,
                s=s,
                excitacao=excitacao,
                gdl=gdl,
                omega_dot=omega_dot,
            )

        return Y[0, :], Y[1, :]

    def plot_resposta_temporal(
            self,
            omega,
            t_inicio=0.0,
            t_fim=1.0,
            dt=1e-4,
            excitacao="livre",
            gdl="ambos",
            omega_rpm=None,
            plot_omega=False,
    ):
        q1, q2 = self.resposta_temporal(
            omega=omega,
            t_inicio=0,
            t_fim=t_fim,
            dt=dt,
            excitacao=excitacao,
            gdl=gdl,
            omega_rpm=omega_rpm
        )

        ## Convertendo para o domínio físico
        u = q1 * self.forma.f(self.eixo.L / 2)
        theta = - q1 * self.forma.g(self.eixo.L / 2)
        v = q2 * self.forma.f(self.eixo.L / 2)
        psi = q2 * self.forma.g(self.eixo.L / 2)

        amp = np.sqrt(u ** 2 + v ** 2)

        if isinstance(omega, Omega):
            t = np.arange(omega.t_0, omega.t_sim + dt, dt)
            if plot_omega:
                omega_list = np.array([omega.v(i) for i in t])
        else:
            t = np.arange(t_inicio, t_fim + dt, dt)
            if plot_omega:
                omega_list = np.ones(len(t)) * omega

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=t,
                y=amp,
                name="Amplitude",
                mode="lines",
                line={
                    "color": "#4787FF",
                    "width": 2,
                },
            ),
            secondary_y=False,
        )
        if plot_omega:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=omega_list,
                    name="Velocidade",
                    mode="lines",
                    line={
                        "color": "#1D1D1D",
                        "width": 1,
                    },
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="Velocidade [RPM]", secondary_y=True)

        fig.update_layout(
            title_text="<b>Resposta Temporal</b>",
            autosize=False,
            width=1200,
            height=600,
        )
        fig.update_xaxes(
            title_text="Tempo [s]",
        )
        fig.update_yaxes(title_text="Amplitude [m]", secondary_y=False)
        return fig

    def plot_resposta(
        self,
        omega: tuple,
        n_points=5000,
        excitacao="desbalanceamento",
        w_rpm=None,
        speed_unit="rpm",
        y_axis_type="log"
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

        if isinstance(omega, Omega):
            t_0 = omega.t_0
            t_f = omega.t_sim

            t_list = np.linspace(t_0, t_f, num=n_points, endpoint=True)
            omega_list = []
            omega_dot_list = []
            for t in t_list:
                omega_list.append(omega.v(t))
                omega_dot_list.append(omega.dot(t))
        else:
            omega_list = np.linspace(
                omega[0], omega[1], num=n_points, endpoint=True
            )

        amplitudes = {}
        amplitude_dict = {
            "amplitude": [],
            "omega": [],
        }

        amplitudes["A1"] = amplitude_dict
        amplitudes["A2"] = amplitude_dict

        cores = ["#4787FF", "#FF4747"]
        fig = go.Figure()
        for i, omega_rpm in enumerate(omega_list):
            if excitacao == "assincrona" and gdl == "separado":
                pass
            else:
                w_rpm = omega_rpm
            if isinstance(omega, Omega):
                amp = np.abs(
                    self.amplitude_vibracao(
                        omega=omega_rpm * (2 * np.pi) / 60,
                        s=s,
                        excitacao=excitacao,
                        gdl=gdl,
                        omega_rpm=w_rpm * (2 * np.pi) / 60,
                        omega_dot=omega_dot_list[i] * (2 * np.pi) / 60,
                    )
                )
            else:
                amp = np.abs(
                    self.amplitude_vibracao(
                        omega=omega_rpm * (2 * np.pi) / 60,
                        s=s,
                        excitacao=excitacao,
                        gdl=gdl,
                        omega_rpm=w_rpm * (2 * np.pi) / 60,
                    )
                )

            amp1 = np.sqrt(amp[0] ** 2 + amp[1] ** 2)
            amp2 = np.sqrt(amp[2] ** 2 + amp[3] ** 2)
            amplitudes["A1"]["amplitude"].append(amp1)
            amplitudes["A2"]["amplitude"].append(amp2)

            freq = omega_rpm if speed_unit == "rpm" else omega_rpm / 60

            amplitudes["A1"]["omega"].append(freq)
            amplitudes["A2"]["omega"].append(freq)

        max_amp = np.max(amplitudes["A1"]["amplitude"])
        idx = 0
        for name, amplitude in amplitudes.items():
            fig.add_trace(
                go.Scatter(
                    x=amplitude["omega"],
                    y=amplitude["amplitude"],
                    mode="lines",
                    name=name,
                    line={
                        "color": cores[idx],
                        "width": 2,
                    },
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
            autosize=False,
            width=1200,
            height=600,
        )
        fig.update_xaxes(
            title=f"Velocidade [{speed_unit}]",
        )
        fig.update_yaxes(
            title="Amplitude [m]",
            # range=[0, max_amp / 50],
            type=y_axis_type,
        )
        return fig

    def plot_campbell(
        self, speed_range: tuple, n_points=100, through="estados", title=None
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
                autosize=False,
                width=1200,
                height=600,
            )

        fig.update_xaxes(
            title="Velocidade [rpm]",
            range=[0, speed_range[1]],
        )
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

        fig = go.Figure()
        color_iterator = iter(tableau_colors)
        max_list = []
        for omega in omega_list:
            omega *= (2 * np.pi) / 60
            if excitacao == "assincrona" and gdl == "separado":
                omega_rpm *= (2 * np.pi) / 60
            else:
                omega_rpm = omega
            A1, B1, A2, B2 = self.amplitude_vibracao(
                omega=omega,
                s=s,
                excitacao=excitacao,
                gdl=gdl,
                omega_rpm=omega_rpm,
            )

            if (A1 * B2) > 0:
                precessao = "FW"
            else:
                precessao = "BW"

            T = (2 * np.pi) / omega
            dt = T / 10
            t = np.linspace(0, T - dt, 100)
            q1 = A1 * np.cos(s * omega * t) + B1 * np.sin(s * omega * t)
            q2 = A2 * np.cos(s * omega * t) + B2 * np.sin(s * omega * t)

            max_list.append(np.max([np.max(q1), np.max(q2)]))

            plot_name = f"Orbita {precessao} - {omega * 60 / (2 * np.pi):.0f} RPM"
            color = next(color_iterator)
            fig.add_trace(
                go.Scatter(
                    x=q1,
                    y=q2,
                    name=plot_name,
                    mode="lines",
                    line={
                        "color": color,
                        "width": 2,
                    },
                    legendgroup=plot_name,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[q1[-1]],
                    y=[q2[-1]],
                    name=plot_name,
                    mode="markers",
                    line={
                        "color": color,
                    },
                    legendgroup=plot_name,
                    showlegend=False,
                )
            )

        max_axis = np.max(max_list)
        fig.update_layout(
            title=f"<b>Órbitas</b> Resposta a(o) {excitacao.capitalize()}",
            autosize=False,
            width=800,
            height=800,
        )
        fig.update_xaxes(
            title=r'$\text{Deslocamento } q_{1}$',
            range=[-max_axis, max_axis]
        )
        fig.update_yaxes(
            title=r'$\text{Deslocamento } q_{2}$',
            range=[-max_axis, max_axis]
        )

        return fig