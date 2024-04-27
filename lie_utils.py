"""
Helper functions for Lie Groups (SO(3) and SE(3))

Copyright (c) 2020 Krishna Murthy Jatavallabhula

Adapted from PointNetLK

https://github.com/hmgoforth/PointNetLK/blob/master/ptlk/sinc.py
https://github.com/hmgoforth/PointNetLK/blob/master/ptlk/so3.py
https://github.com/hmgoforth/PointNetLK/blob/master/ptlk/se3.py

Reproducing PointNetLK License

MIT License

Copyright (c) 2019 Hunter Goforth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import cos, sin
from pkm.util.math_util import skew_matrix


def get_small_and_large_angle_inds(theta: torch.Tensor, eps: float = 1e-3):
    r"""Returns the indices of small and non-small (large) angles, given
    a tensor of angles, and the threshold below (exclusive) which angles
    are considered 'small'.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.
    """

    small_inds = torch.abs(theta) < eps
    large_inds = torch.logical_not(small_inds)

    return small_inds, large_inds


def sin_theta_by_theta(theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{sin \theta}{\theta}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.
    """
    return torch.sinc(theta / torch.pi)

    # Tensor to store result.
    result = torch.zeros_like(theta)

    small_inds, large_inds = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta[small_inds] ** 2
    result[small_inds] = 1 - theta_sq / 6 * (
        1 - theta_sq / 20 * (1 - theta_sq / 42))

    # For large angles, compute using torch.sin and torch.cos
    result[large_inds] = torch.sinc(theta[large_inds] / torch.pi)
    # result[large_inds] = torch.sin(theta[large_inds]) / theta[large_inds]

    return result


def grad_sin_theta_by_theta(theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{\partial sin \theta}{\partial \theta \theta}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold below which an angle is considered small.

    """

    # Tensor to store result.
    result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta ** 2
    result[s] = (-theta[s] / 3 *
                 (1 - theta_sq[s] / 10 * (1 - theta_sq[s] / 28 * (1 - theta_sq[s] / 54))))

    # For large angles, compute using torch.sin and torch.cos
    result[l] = cos(theta[l]) / theta[l] - sin(theta[l]) / theta_sq[l]

    return result


def grad_sin_theta_by_theta_div_theta(theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{\partial sin \theta}{\partial \theta \theta} \frac{1}{\theta}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold below which an angle is considered small.

    """

    # Tensor to store result.
    result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta ** 2
    result[s] = (-1 / 3 *
                 (1 - theta_sq[s] / 10 * (1 - theta_sq[s] / 28 * (1 - theta_sq[s] / 54))))

    # For large angles, compute using torch.sin and torch.cos
    result[l] = (cos(theta[l]) / theta[l] - sin(theta[l]) / theta_sq[l]) / theta[l]

    return result


# def one_minus_cos_theta_by_theta_sq(theta: torch.Tensor, eps: float = 1e-3):
#     r"""Computes :math:`\frac{\theta}{sin \theta}`.

#     Args:
#         theta (torch.Tensor): Angle (magnitude of axis-angle vector).
#         eps (float): Threshold (exclusive) below which an angle is
#             considered 'small'.

#     """

#     # Tensor to store result.
#     result = torch.zeros_like(theta)

#     s, l = get_small_and_large_angle_inds(theta, eps)

#     # Use Taylor series approximation for small angles
#     # (upto powers O(theta**8)).
#     if False:
#         theta_sq = theta[s] ** 2
    
#         result[s] = (((31 * theta_sq) / 42 + 7)
#                     * theta_sq / 60 + 1) * theta_sq / 6 + 1

#         # For large angles, compute using torch.sin and torch.cos
#         result[l] = theta[l] / torch.sin(theta[l])
#     else:
#         # theta_sq = theta[s] ** 2
#         # result[s] = 1/2*(1-theta_sq[s]/12*(1-theta_sq[s]/30*(1-theta_sq[s]/56)))
#         # result[l] = (1-cos(theta[l]))/theta_sq[l]
#         theta_sq = torch.square(theta)
#         result = torch.where(s,
#                              1/2*(1-theta_sq/12*(1-theta_sq/30*(1-theta_sq/56))),
#                              (1-torch.cos(theta))/theta_sq)
                          

#     return result


def grad_one_minus_cos_theta_by_theta_sq(
        theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{\partial \theta}{\partial \theta sin \theta}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold below which an angle is considered small.

    """

    # Tensor to store result.
    result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta[s] ** 2
    result[s] = (
        ((((127 * theta_sq) / 30 + 31) * theta_sq / 28 + 7) * theta_sq / 30 + 1) *
        theta[s] / 3)

    # For large angles, compute using torch.sin and torch.cos
    result[l] = 1 / sin(theta[l]) - (theta[l] * cos(theta[l])) / (
        sin(theta[l]) * sin(theta[l])
    )

    return result


def grad_one_minus_cos_theta_by_theta_sq_div_sin_theta(
    theta: torch.Tensor, eps: float = 1e-3
):
    r"""Computes :math:`\frac{\partial \theta}{\partial \theta sin \theta} \frac{1}{sin \theta}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold below which an angle is considered small.

    """

    # Tensor to store result.
    result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta[s] ** 2
    result[s] = theta_sq * (theta_sq * ((4 * theta_sq) / 675 + 2 / 63) + 2 / 15) + 1 / 3

    # For large angles, compute using torch.sin and torch.cos
    result[l] = (1 / sin(theta[l]) - (theta[l] * cos(theta[l])) /
                 (sin(theta[l]) * sin(theta[l]))) / sin(theta[l])

    return result


def one_minus_cos_theta_by_theta_sq(theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{1 - cos \theta}{\theta^2}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """

    # Tensor to store result.
    # result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    # theta_sq = theta ** 2
    # result[s] = (
    #     1 / 2 *
    #     (1 - theta_sq[s] / 12 * (1 - theta_sq[s] / 30 * (1 - theta_sq[s] / 56))))

    # # For large angles, compute using torch.sin and torch.cos
    # result[l] = (1 - cos(theta[l])) / theta_sq[l]
    
    theta_sq = torch.square(theta)

    rsmall = (-(theta_sq**3 - 56*theta_sq**2 + 1680*theta_sq - 20160)/40320)
    # rsmall = 1/2*(1-theta_sq/12*(1-theta_sq/30*(1-theta_sq/56)))
    result = torch.where(s, rsmall, (1-torch.cos(theta))/theta_sq)

    return result


def grad_one_minus_cos_theta_by_theta_sq(
        theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{\partial}{\partial \theta}\frac{1 - cos \theta}{\theta^2}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """

    # Tensor to store result.
    result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta ** 2
    result[s] = (-theta[s] / 12 *
                 (1 - theta_sq[s] / 5 *
                  (1 / 3 - theta_sq[s] / 56 * (1 / 2 - theta_sq[s] / 135))))

    # For large angles, compute using torch.sin and torch.cos
    result[l] = sin(theta[l]) / theta_sq[l] - 2 * (1 - cos(theta[l])) / (
        theta_sq[l] * theta[l]
    )

    return result


def theta_minus_sin_theta_by_theta_cube(
        theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{\theta - sin \theta}{\theta^3}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """

    # Tensor to store result.
    result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta[s] ** 2
    result[s] = 1 / 6 * (1 - theta_sq / 20 *
                         (1 - theta_sq / 42 * (1 - theta_sq / 72)))

    # For large angles, compute using torch.sin and torch.cos
    result[l] = (theta[l] - sin(theta[l])) / (theta[l] ** 3)

    return result


def grad_theta_minus_sin_theta_by_theta_cube(
        theta: torch.Tensor, eps: float = 1e-3):
    r"""Computes :math:`\frac{\partial}{\partial \theta}\frac{\theta - sin \theta}{\theta^3}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """

    # Tensor to store result.
    result = torch.zeros_like(theta)

    s, l = get_small_and_large_angle_inds(theta, eps)

    # Use Taylor series approximation for small angles
    # (upto powers O(theta**8)).
    theta_sq = theta[s] ** 2
    result[s] = (
        -theta[s]
        / 60
        * (1 - theta_sq / 21 * (1 - theta_sq / 24 * (1 / 2 - theta_sq / 165)))
    )

    # For large angles, compute using torch.sin and torch.cos
    result[l] = (3 * sin(theta[l]) - theta[l]
                 * (cos(theta[l]) + 2)) / (theta[l] ** 4)

    return result


class SinThetaByTheta_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return sin_theta_by_theta(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_sin_theta_by_theta(theta).to(
                grad_output.device
            )
        return grad_theta


class SinThetaByTheta(torch.nn.Module):
    def __init__(self):
        super(SinThetaByTheta, self).__init__()

    def forward(self, x):
        return SinThetaByTheta_Function.apply(x)


class ThetaBySinTheta_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return one_minus_cos_theta_by_theta_sq(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_one_minus_cos_theta_by_theta_sq(
                theta).to(
                grad_output.device)
        return grad_theta


class ThetaBySinTheta(torch.nn.Module):
    def __init__(self):
        super(ThetaBySinTheta, self).__init__()

    def forward(self, x):
        return ThetaBySinTheta_Function.apply(x)


class OneMinusCosThetaByThetaSq_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return one_minus_cos_theta_by_theta_sq(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_one_minus_cos_theta_by_theta_sq(
                theta).to(
                grad_output.device)
        return grad_theta


class OneMinusCosThetaByThetaSq(torch.nn.Module):
    def __init__(self):
        super(OneMinusCosThetaByThetaSq, self).__init__()

    def forward(self, x):
        return OneMinusCosThetaByThetaSq_Function.apply(x)


class ThetaMinusSinThetaByThetaCube_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return theta_minus_sin_theta_by_theta_cube(theta)

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_theta_minus_sin_theta_by_theta_cube(
                theta).to(grad_output.device)
        return grad_theta


class ThetaMinusSinThetaByThetaCube(torch.nn.Module):
    def __init__(self):
        super(ThetaMinusSinThetaByThetaCube, self).__init__()

    def forward(self, x):
        return ThetaMinusSinThetaByThetaCube_Function.apply(x)


# Initialize the coefficient objects, for reuse
coeff_A = SinThetaByTheta()
coeff_B = OneMinusCosThetaByThetaSq()
coeff_C = ThetaMinusSinThetaByThetaCube()


class SO3:
    def __init__(self):
        pass

    @staticmethod
    def cross_product(x: torch.Tensor, y: torch.Tensor):
        return torch.cross(x.view(-1, 3), y.view(-1, 3), dim=1).view_as(x)

    @staticmethod
    def liebracket(x, y):
        return SO3.cross_product(x, y)

    @staticmethod
    def hat(x):
        # size: [B, 3] -> [B, 3, 3]
        x_ = x.view(-1, 3)
        x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
        O = torch.zeros_like(x1)

        X = torch.stack(
            (
                (torch.stack((O, -x3, x2), dim=1)),
                (torch.stack((x3, O, -x1), dim=1)),
                (torch.stack((-x2, x1, O), dim=1)),
            ),
            dim=1,
        )
        return X.view(*(x.size()[0:-1]), 3, 3)

    @staticmethod
    def vee(X):
        # size: [B, 3, 3] -> [B, 3]
        X_ = X.view(-1, 3, 3)
        x1, x2, x3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
        x = torch.stack((x1, x2, x3), dim=1)
        return x.view(*X.size()[0:-2], 3)

    @staticmethod
    def genvec(*args, **kwds):
        return torch.eye(3, *args, **kwds)

    @staticmethod
    def genmat(*args, **kwds):
        return SO3.hat(SO3.genvec(*args, **kwds))

    @staticmethod
    def Exp(x):
        # Exp map
        omega = x.view(-1, 3)
        theta = omega.norm(p=2, dim=1).view(-1, 1, 1)
        omega_hat = SO3.hat(omega)
        omega_hat_sq = omega_hat.bmm(omega_hat)
        I = torch.eye(3).to(omega.device)
        R = I + coeff_A(theta) * omega_hat + coeff_B(theta) * omega_hat_sq
        return R.view(*(x.size()[0:-1]), 3, 3)

    @staticmethod
    def inverse(g):
        # input: group element [B x 3 x 3]
        # output: group element [B x 3 x 3]
        R = g.view(-1, 3, 3)
        Rt = R.transpose(1, 2)
        return Rt.view_as(g)

    @staticmethod
    def btrace(X):
        # Batch trace: [B, N, N] -> [B]
        n = X.size(-1)
        X_ = X.view(-1, n, n)
        tr = torch.zeros(X_.size(0)).to(X.device)
        for i in range(tr.size(0)):
            m = X_[i, :, :]
            tr[i] = torch.trace(m)
        return tr.view(*(X.size()[0:-2]))

    @staticmethod
    def Log(g):
        # Log map
        # input: group element [B x 3 x 3]
        # output: tangent space element [B x 3 x 3]
        eps = 1e-7
        R = g.view(-1, 3, 3)
        tr = SO3.btrace(R)
        c = (tr - 1) / 2
        t = torch.acos(c)
        sc = coeff_A(t)
        idx0 = torch.abs(sc) <= eps
        idx1 = torch.abs(sc) > eps
        sc = sc.view(-1, 1, 1)

        X = torch.zeros_like(R)
        if idx1.any():
            X[idx1] = (R[idx1] - R[idx1].transpose(1, 2)) / (2 * sc[idx1])

        if idx0.any():
            t2 = t[idx0] ** 2
            A = (R[idx0] + torch.eye(3).type_as(R).unsqueeze(0)
                 ) * t2.view(-1, 1, 1) / 2
            aw1 = torch.sqrt(A[:, 0, 0])
            aw2 = torch.sqrt(A[:, 1, 1])
            aw3 = torch.sqrt(A[:, 2, 2])
            sgn_3 = torch.sign(A[:, 0, 2])
            sgn_3[sgn_3 == 0] = 1
            sgn_23 = torch.sign(A[:, 1, 2])
            sgn_23[sgn_23 == 0] = 1
            sgn_2 = sgn_23 * sgn_3
            w1 = aw1
            w2 = aw2 * sgn_2
            w3 = aw3 * sgn_3
            w = torch.stack((w1, w2, w3), dim=-1)
            omega_hat = SO3.hat(w)
            X[idx0] = omega_hat

        x = SO3.vee(X.view_as(g))
        return x

    @staticmethod
    def inv_vecs_Xg_ig(x):
        r""" H = inv(vecs_Xg_ig(x)) """
        t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
        X = SO3.hat(x)
        S = X.bmm(X)
        I = torch.eye(3).to(x)

        s, l = get_small_and_large_angle_inds(t)
        eta = torch.zeros_like(t)
        t2 = t[s] ** 2
        eta[s] = ((t2 / 40 + 1) * t2 / 42 + 1) * t2 / 720 + 1 / 12
        eta[l] = (1 - (t[l] / 2) / torch.tan(t[l] / 2)) / (t[l] ** 2)
        H = I - 1 / 2 * X + eta * S
        return H.view(*(x.size()[0:-1]), 3, 3)


class SO3Exp_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        g = SO3.Exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        g = SO3.Exp(x)
        gen_k = SO3.genmat().to(x.device)

        dg = gen_k.matmul(g.view(-1, 1, 3, 3))
        dg = dg.to(grad_output.device)
        go = grad_output.contiguous().view(-1, 1, 3, 3)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input


class SE3:

    # Each SE3 element is [w, v] -> [rotation, translation]

    def __init__(self):
        pass

    @staticmethod
    def twist_product(x, y):
        x_ = x.view(-1, 6)
        y_ = y.view(-1, 6)

        xw, xv = x_[:, 0:3], x_[:, 3:6]
        yw, yv = y_[:, 0:3], y_[:, 3:6]

        zw = SO3.cross_product(xw, yw)
        zv = SO3.cross_product(xw, yv) + SO3.cross_product(xv, yw)

        z = torch.cat((zw, zv), dim=1)

        return z.view_as(x)

    @staticmethod
    def liebracket(x, y):
        return SE3.twist_product(x, y)

    @staticmethod
    def hat(x):
        # size: [B, 6] -> [B, 4, 4]
        x_ = x.view(-1, 6)
        w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
        v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
        O = torch.zeros_like(w1)

        X = torch.stack(
            (
                torch.stack((O, -w3, w2, v1), dim=1),
                torch.stack((w2, O, -w1, v2), dim=1),
                torch.stack((-w2, w1, O, v3), dim=1),
                torch.stack((O, O, O, O), dim=1),
            ),
            dim=1,
        )
        return X.view(*(x.size()[0:-1]), 4, 4)

    @staticmethod
    def vee(X):
        # size: [B, 4, 4] -> [B, 6]
        X_ = X.view(-1, 4, 4)
        w1, w2, w3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
        v1, v2, v3 = X_[:, 0, 3], X_[:, 1, 3], X_[:, 2, 3]
        x = torch.stack((w1, w2, w3, v1, v2, v3), dim=1)
        return x.view(*X.size()[0:-2], 6)

    @staticmethod
    def genvec(*args, **kwds):
        return torch.eye(6, *args, **kwds)

    @staticmethod
    def genmat(*args, **kwds):
        return SE3.hat(SE3.genvec(*args, **kwds))

    @staticmethod
    # @torch.jit.script
    @torch.compile
    def Exp(x:torch.Tensor)->torch.Tensor:
        x_ = x.view(-1, 6)
        w, v = x_[:, 0:3], x_[:, 3:6]
        # t = w.norm(p=2, dim=1).view(-1, 1, 1)
        # print('w', w.shape)
        #tsq = torch.tensordot(w, w, dims=( 1,1 ))

        # squared norm of `w`
        tsq = torch.einsum('...i, ...i -> ...', w, w).view(-1,1,1)

        #W = SO3.hat(w)
        W = skew_matrix(w)
        I = torch.eye(3, device=w.device)#.to(w)

        # Would like to avoid this sqrt if possible
        t   = tsq.sqrt()

        # Would like to avoid
        # this reciprocal if possible...!
        itsq = (tsq + 1e-6).reciprocal()

        # A = sinc(t) i.e. sin(t)/t
        S = W.bmm(W) * itsq
        # S2 = torch.einsum('...i, ...j -> ...ij', w, w) * itsq
        # S2 = torch.linalg.cross(w,w)
        # print('sms2', S-S2)

        # A = c * torch.eye(cls.dim, dtype=phi.dtype)
        # B = (1. - c) * utils.outer(axis, axis)
        # C = s * cls.wedge(axis)

        # S = S * itsq
        A = torch.sinc(t / torch.pi)
        B = (1 - torch.cos(t))
        C = (1 - A)

        # R = I + AW + BWW
        # V = I + BW + (1-A)WW
        R = I + A * W + B * S
        V = I + B * W * itsq + C * S
        # = I + BW/t2 + (1-A)WW
        # = 

        p = V.bmm(v.reshape(-1, 3, 1))
        # z = (
        #     torch.as_tensor([0, 0, 0, 1], device=x.device)
        #     .view(1, 1, 4)
        #     .repeat(x_.size(0), 1, 1)
        #     # .to(x.device)
        # )
        Rp = torch.cat((R, p), dim=2)
        # g0 = torch.cat((Rp, z), dim=1)
        g = torch.nn.functional.pad(Rp, (0,0, 0,1),
                                    'constant', 0.0)
        g[..., 3, 3] = 1
        # g1 = g
        # print(g0-g1)
        # return g.view(*(x.size()[0:-1]), 4, 4)
        return g.view(x.shape[:-1] + (4,4))

    @staticmethod
    def inverse(g):
        g_ = g.view(-1, 4, 4)
        R = g_[:, 0:3, 0:3]
        p = g_[:, 0:3, 3]
        Q = R.transpose(1, 2)
        q = -Q.matmul(p.unsqueeze(-1))
        z = (
            torch.Tensor([0, 0, 0, 1])
            .view(1, 1, 4)
            .repeat(g_.size(0), 1, 1)
            .to(g.device)
        )
        Qq = torch.cat((Q, q), dim=2)
        ig = torch.cat((Qq, z), dim=1)
        return ig.view(*(g.size()[0:-2]), 4, 4)

    @staticmethod
    def Log(g):
        g_ = g.view(-1, 4, 4)
        R = g_[:, 0:3, 0:3]
        p = g_[:, 0:3, 3]
        w = SO3.Log(R)
        H = SO3.inv_vecs_Xg_ig(w)
        v = H.bmm(p.contiguous().view(-1, 3, 1)).view(-1, 3)
        x = torch.cat((w, v), dim=1)
        return x.view(*(g.size()[0:-2]), 6)


class SE3Exp_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        # ctx.save_for_backward(x)
        # return SE3.Exp(x)
        y = SE3.Exp(x)
        ctx.save_for_backward(x, y, g)
        # ctx.save_for_misc = oe_expr
        return y

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        (x, y, g) = ctx.saved_tensors
        # oe_expr = ctx.save_for_misc
        
        # g = SE3.Exp(x)
        # gen_k = SE3.genmat(device=x.device)#.to(x.device)
        # for i in range(6):
        #     print(gen_k[i])
        # print(gen_k.shape)

        # dg = gen_k.matmul(y.view(-1, 1, 4, 4))
        # dg = dg.to(grad_output.device)
        # print(dg)

        # go = grad_output#.contiguous().view(-1, 1, 4, 4)

        # 64,1,4,4 ; 64,6,4,4
        # grad_input = oe_expr(grad_output.reshape(-1,4,4),
        #                      y.view(-1,4,4))

        # print(x[0])
        # print(y.view(-1,4,4)[0])
        # print(torch.einsum('ijl,...lk -> ...ijk',
        #                    g, y.view(-1,4,4))[0])

        grad_input = torch.einsum(
                # '...jk,...ijk->...i',
                '...jk,ijl,...lk -> ...i',
                                  grad_output.reshape(-1,4,4),
                                  # dg
                                  g,
                                  y.view(-1,4,4))
        # dd = go.reshape(-1, 4, 4) * dg
        # grad_input = dd.sum(dim=(-1,-2))#.sum(-1)
        # print(go.reshape(-1, 1, 4, 4).shape, dg.shape)
        # grad_input = th.einsum('
        return grad_input, None


SO3Exp = SO3Exp_Function.apply
SE3Exp = SE3Exp_Function.apply


# class LocalProjection(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, T, se3, x):
#         ctx.save_for_backward(T, x)
#         return T @ SE3.Exp(se3) @ x

#     @staticmethod
#     def backward(ctx, grad_output):
#         T, x = ctx.saved_tensors
#         th.cat([T[..., :3, :3], skew_matrix(x)],
#                dim
#         (theta,) = ctx.saved_tensors
#         grad_theta = None
#         if ctx.needs_input_grad[0]:
#             grad_theta = grad_output * grad_sin_theta_by_theta(theta).to(
#                 grad_output.device
#             )
#         return grad_theta

def main():
    import torch as th
    from pkm.util.math_util import random_quat, axa_from_quat
    r = axa_from_quat(random_quat(size=5))
    print( SO3.Exp(r) )
    t = th.randn(size=(5, 3))
    rt = th.cat([r, t], dim=-1)
    T = SE3Exp(rt)
    print(T)
    pass


if __name__ == "__main__":
    main()
