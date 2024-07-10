### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ fd4fa4bb-c34b-4557-b17a-57b8fc3888d3
using PlutoUI

# ╔═╡ 3015a3a0-0469-4ffe-b909-d28ddbbd22b4
html"""
<style>
body:not(.disable_ui) main {
	max-width: 95%;
	margin-right: 10px;
	align-self: center;
}
</style>
"""

# ╔═╡ 41db52a5-812e-49bb-9d1f-fb7851740a88
md"""
# Eigenvalue algorithms for matrices of quaternions and reduced biquaternions and applications 

##### by Ivan Slapničar, Nevena Jakovčević Stor, Anita Carević and Thaniporn Chaysri from the University of Split, FESB, and Sk. Safique Ahmad and Neha Bhadala from the IIT Indore, Department of Mathematics

##### at the 9-th European Mathematical Congress, Sevilla, July 15-19, 2024.

This work has been partially supported by the Croatian Science Foundation under the project IP-2020-02-2240 - [http://manaa.fesb.unist.hr](http://manaa.fesb.unist.hr).

$(PlutoUI.LocalResource("./HRZZ-eng-170x80-1.jpg"))

"""

# ╔═╡ 0f0099d9-5183-4aea-accb-2ae4e088172d
md"
# Aims
* To state basic NLA problems and corresponding (generic) algorithms
* To show how to implement algorithms for quaternions and reduced biquaternions
* To give some interesting examples
"

# ╔═╡ 9bbeb6fc-3500-4fef-8e26-758b20b581b8
md"
# Basic LA problems and algorithms

* Solving systems - Gaussian elimination $PA=LU$ _(not generic!)_
* Least squares (also for systems) - QR factorization $A=QR$ or $AP=QR$  
  * (Householder) `reflector!()`
  * `reflectorApply!()`
* Eigenvalues 
  * reduction to Hessenberg form $X^* A X=H$ using reflectors
  * reduction to Schur form $U^*HU=T$ using pivots and `givens()` rotations
  * Then $Q=XU$ and $Q^*AQ=T$
* Singular values
  * reduction to bidiagonal form $X^* A Y=B$ using reflectors
  * bidiagonal SVD $Q^*BR=\Sigma$ using pivots and `givens()` rotations
  * Then $U=XQ$, $V=YR$ and $U^*AV=\Sigma$

"


# ╔═╡ 215b1cbf-5c80-4229-914e-f830e9a19150
md"
# Compute the reflector
From Julia's package `LinearAlgebra`, file `generic.jl`
```jldoctest
# Elementary reflection similar to LAPACK. The reflector is not Hermitian but
# ensures that tridiagonalization of Hermitian matrices become real. See lawn72

@inline function reflector!(x::AbstractVector{T}) where {T}
    require_one_based_indexing(x)
    n = length(x)
    n == 0 && return zero(eltype(x))
    @inbounds begin
        ξ1 = x[1]
        normu = norm(x)
        if iszero(normu)
            return zero(ξ1/normu)
        end
        ν = T(copysign(normu, real(ξ1)))
        ξ1 += ν
        x[1] = -ν
        for i = 2:n
            x[i] /= ξ1
        end
    end
    ξ1/ν
end
```
"

# ╔═╡ caccc0ce-fdec-461a-89f3-4a14ba1f93eb
md"""
# Apply the reflector - generic
From Julia's package `LinearAlgebra`, file `generic.jl`

"Generic" means that this function can be used for __ANY__ number system.
```jldoctest
#    reflectorApply!(x, τ, A)
#
# Multiplies `A` in-place by a Householder reflection on the left. 
# It is equivalent to `A .= (I - conj(τ)*[1; x] * [1; x]')*A`.

@inline function reflectorApply!(x::AbstractVector, τ::Number, A::AbstractVecOrMat)
    require_one_based_indexing(x)
    m, n = size(A, 1), size(A, 2)
    if length(x) != m
        throw(DimensionMismatch(lazy"reflector has length $(length(x)), which must match the first dimension of matrix A, $m"))
    end
    m == 0 && return A
    @inbounds for j = 1:n
        Aj, xj = view(A, 2:m, j), view(x, 2:m)
        vAj = conj(τ)*(A[1, j] + dot(xj, Aj))
        A[1, j] -= vAj
        axpy!(-vAj, xj, Aj)
    end
    return A
end
```
"""

# ╔═╡ 373c203c-c624-45a7-a7ea-abc79f2b1a48
md"""
# "Plain" QR factorization - generic
From Julia's package `LinearAlgebra`, file `qr.jl`
```jlddoctest
function qrfactUnblocked!(A::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    τ = zeros(T, min(m,n))
    for k = 1:min(m - 1 + !(T<:Real), n)
        x = view(A, k:m, k)
        τk = reflector!(x)
        τ[k] = τk
        reflectorApply!(x, τk, view(A, k:m, k + 1:n))
    end
    QR(A, τ)
end
```

The function `qrfactPivotedUnblocked!(A::AbstractMatrix)` from the same file is generic, too.
"""

# ╔═╡ 8cc9e8f5-2a27-4288-af7c-8e6d53276693
md"""
# Givens rotations
From Julia's package `LinearAlgebra`, file `qr.jl`
```jldoctest
function givens(f::T, g::T, i1::Integer, i2::Integer) where T
    if i1 == i2
        throw(ArgumentError("Indices must be distinct."))
    end
    c, s, r = givensAlgorithm(f, g)
    if i1 > i2
        s = -conj(s)
        i1, i2 = i2, i1
    end
    Givens(i1, i2, c, s), r
end
```
This function is generic. The only non-generic part is `givensAlgorithm(f, g)`.
It must be implemented with care (see BLAS for $\mathbb{R}$ and $\mathbb{C}$).
"""

# ╔═╡ 2552eeb1-c631-4e58-94f1-3894b9545ab9
md"
# Quaternions - $\mathbb{Q}$

Quaternions are a non-commutative associative number system that extends complex numbers (a four-dimensional non-commutative algebra and a division ring of numbers), introduced by Hamilton ( [1853](https://openlibrary.org/books/OL23416635M/Lectures_on_quaternions), [1866](https://openlibrary.org/books/OL7211578M/Elements_of_quaternions.)). Basis elements are $1$, $\mathbf {i}$, 
$\mathbf {j}$, and $\mathbf {k}$, satisfying the formula 

$$
\displaystyle \mathbf {i} ^{2}=\mathbf {j} ^{2}=\mathbf {k} ^{2}=\mathbf {i\,j\,k} =-1.$$

Quaternion $q\in\mathbb{Q}$ has the form

$$q=a+b\ \mathbf {i} +c\ \mathbf {j} +d\ \mathbf {k},\quad a,b,c,d, \in \mathbb{R}.$$

Quaternions $p$ and $q$ are __similar__ if $p=x^{-1} q x$ for some quaternion $x$.

The __standard form__ of the quaternion $q$ is the unique similar quaternion $q_s=x^{-1} qx =a + \hat b\,  \mathbf{i}$, where $\|x\|=1$ and $\hat b \geq 0$.

([Sudbery,1979](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/quaternionic-analysis/308CF454034EC347D4D17D1F829F8471)) The value of a complex analytic function $f$ at $q\in\mathbb{Q}$, is computed by evaluating the extension of $f$ to the quaternions at $q$ , for example,

$$\sqrt{q}=\pm \left(\sqrt {\frac {\|q\|+a_1}{2}} + \frac {\operatorname{imag} (q)}{\|\operatorname{imag}(q)\|} \sqrt {\frac {\|q\|-a_1}{2}}\right).$$

Basic operations and computation of functions are implemented in the package [Quaternions.jl](https://github.com/JuliaGeometry/Quaternions.jl).
"

# ╔═╡ 758580cc-ad78-4ce2-a427-16c014303d11
md"
# Reduced Bi-Quaternions - $\mathbb{Q}_\mathbb{R}$

Reduced Bi-Quaternions are a _commutative_ associative number system that extends complex numbers, introduced by Segre ( [1892](https://link.springer.com/article/10.1007/BF01443559)). Basis elements are $1$, $\mathbf {i}$, 
$\mathbf {j}$, and $\mathbf {k}$, satisfying formulas 

$$
\displaystyle \mathbf {i} ^{2}=-\mathbf {j} ^{2}=\mathbf {k} ^{2}=-1, \quad 
\mathbf{i}\mathbf{j}=\mathbf{j}\mathbf{i}=\mathbf{k},\quad 
\mathbf{j}\mathbf{k}=\mathbf{k}\mathbf{j}=\mathbf{i},\quad
\mathbf{k}\mathbf{i}=\mathbf{i}\mathbf{k}=-\mathbf{i}.$$

Basic non-trivial __zero divisors__ are $\displaystyle e_1=\frac{1+\mathbf{j}}{2}$ and $\displaystyle e_2=\frac{1-\mathbf{j}}{2}$, 

$$
e_1\cdot e_1=e_1,\quad e_2\cdot e_2=e_2,\quad e_1\cdot e_2=0.$$

Any $a = a_0 + a_1 \mathbf{i} + a_2 \mathbf{j} + a_3 \mathbf{k} \in\mathbb{Q}_\mathbb{R}$ is a linear combination of
$e_1$ and $e_2$ (the __splitting__):

$$
a=a_{c1}e_1+a_{c2}e_2=[a_1+a_2+\mathbf{i}(a_1+a_3)]e_1+[a_1-a_2+\mathbf{i}(a_1-a_3)]e_2.$$

The splittings are defined analogously for vectors and matrices. 

Basic operations are implemented in the package [RBiQuaternions.jl](https://github.com/ivanslapnicar/RBiQuaternions.jl).

"

# ╔═╡ 75268c27-c93e-45af-8cec-d3c1e74f4b7c
md"
# Conjugation and norm

For $q\in\mathbb{Q}$, the __conjugation__ is defined by 

$$\bar q=a-b\ \mathbf {i} -c\ \mathbf {j} -d\ \mathbf {k},$$

and the __norm__ is defined by (quaternions are a Hilbert space), 

$$\bar q q=q\bar q=|q|^2=\|q\|^2=a^2+b^2+c^2+d^2.$$

For $a = a_0 + a_1 \mathbf{i} + a_2 \mathbf{j} + a_3 \mathbf{k} \in\mathbb{Q}_\mathbb{R}$, the __conjugation__ is defined by 

$$\bar a=a_0-a_1\ \mathbf {i} +a_2\ \mathbf {j} -a_3\ \mathbf {k},$$

and the __norm__ is defined by

$$|a|^2=\|a\|^2=a_0^2+a_1^2+a_2^2+a_3^2.$$

In all cases, the dot product of two vectors, $a$ and $y$ is defined as

$$
x\cdot y=x^*y= \sum \overline{x_i}\ y_i.$$
"

# ╔═╡ fa9e44d2-5729-45c1-9aa3-a2d35372aebd
md"
# Homomorphisms

Quaternions are homomorphic to $\mathbb{C}^{2\times 2}$:

$$
\mathbb{Q}\ni q\to \begin{bmatrix}a+b\,\mathbf{i} & c+d\, \mathbf{i}\\-c+d\, \mathbf{i} & a-b\, \mathbf{i}\end{bmatrix}\equiv C(q),$$

with eigenvalues $q_s$ and $\bar q_s$. It holds

$$
C(p+q)=C(p)+C(q),\quad C(pq)=C(p)C(q)\quad C(\bar p)=\overline{C(p)}.$$

Reduced bi-quaternions are homomorphic to complex symmetric matrices from $\mathbb{C}^{2\times 2}$ (zero divisors!):

$$
\mathbb{Q}_\mathbb{R}\ni a\to \begin{bmatrix}a_0-a_1\,\mathbf{i} & a_2-a_3\, \mathbf{i}\\a_2-a_3\, \mathbf{i} & a_0-a_1\, \mathbf{i}\end{bmatrix}\equiv C(a),$$

Again

$$
C(a+b)=C(a)+C(b),\quad C(ab)=C(a)C(b)\quad C(\bar a)=\overline{C(a)}.$$


"

# ╔═╡ b299fcf7-7ced-45d1-a55c-74482ecb0c60
md"
# Eigenvalue decomposition in $\mathbb{Q}^{n\times n}$

Right eigenpairs $(λ,x)$ satisfy

$$
Ax=xλ, \quad x\neq 0.$$

Usually, $x$ is chosen such that $\lambda$ is the standard form.

Eigenvalues are invariant under similarity.

> Eigenvalues are __NOT__ shift invariant, that is, eigenvalues of the shifted matrix are __NOT__ the shifted eigenvalues. (In general, $X^{-1}qX\neq qX^{-1}X=qI$)

If $\lambda$ is in the standard form, it is invariant under similarity with complex numbers.  
"

# ╔═╡ 82c8a221-3125-42a1-a033-02408688b6ae
md"""
# A Quaternion QR algorithm

_by Angelika Bunse-Gerstner, Ralph Byers, and Volker Mehrmann, Numer. Math 55, 83-95 (1989)_

* native functions `reflector!()` and `reflectorApply!()` work for quaternions as is
* native function `hessenberg()` from the package `GenericLinearAlgebra.jl` works as is
* Schur factorization requires quaternion implementation of `givens()`:
```jldoctest
function givensAlgorithm(f::T, g::T) where T<:Quaternion
    if f==zero(T)
        return zero(T), one(T), abs(g)
    else
        t=g/f
        cs=abs(f)/hypot(f,g)
        sn=t'*cs
        r=cs*f+sn*g
        return cs,sn,r
    end
end
```
The algorithm is derived for general matrices and requires $O(n^3)$ operations. The algorithm is stable.
"""

# ╔═╡ 02afde06-04e6-44ae-b8c4-3e8e00d5b7cc
md"""
#  Computing the Schur decomposition

Given the upper Hessenberg matrix $A \in\mathbb{Q}^{n\times n}$, the method 
applies complex shift $\mu$ to $A$ by using Francis standard double shift on the matrix 

$$
M=A^2-(\mu+\bar\mu)A+\mu\bar\mu I$$

and applying it implicitly on $A$.

If $Ax=x\lambda$, then 

$$
\begin{aligned}
Mx&=(A^2-(\mu+\bar\mu)A+\mu\bar\mu I)x=x\lambda^2-
x(\mu+\bar\mu)\lambda+x\mu\bar\mu\\
&=x(\lambda^2-(\mu+\bar\mu)\lambda+\mu\bar\mu)
\end{aligned}$$

For the perfect shift, $\mu=\lambda$, it holds

$$
\lambda^2-(\mu+\bar\mu)\lambda+\mu\bar\mu=\lambda^2-(\lambda+\bar\lambda)\lambda+\lambda\bar\lambda=0.$$

Details are given in Algorithm 4 in the Appendix of [BGBM89].
"""

# ╔═╡ 0b1cd856-9f43-4e2f-bade-ad01df6aee0e
md"""
# Perturbation analysis

We have the following Bauer-Fike type theorem (_Sk. Safique Ahmad, Istkhar Ali, and Ivan Slapničar,\
Perturbation analysis of matrices over a quaternion division algebra, ETNA, Volume 54, pp. 128-149, 2021._):

Let $A=X\Lambda X^{-1}$, where $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$ and $\lambda_i$ is in standard form. 
If $\mu$ is  a standard right eigenvalue of $A+\Delta A$, then 

$$
\operatorname{dist}(\mu, \Lambda_s(A)) = \min_{\lambda_i \in \Lambda_s(A)} \{ | \lambda_i  - \mu | \} \leq \kappa (X) \| \Delta A \|_2.$$

The residual bound is as follows: let $(\tilde \lambda,\tilde x)$ be the approximate eigenpair of the matrix $A$, where $\|\tilde x\|_2=1$, and let

$$
r=A\tilde x-\tilde x\tilde \lambda,\quad \Delta A=-{r\tilde x^*}.$$

Then, $(\tilde \lambda,\tilde x)$ is the eigenpair of the matrix $A+\Delta A$ and 
$\|\Delta A\|_2\leq \|r\|_2$.
"""

# ╔═╡ ba519f07-3941-4144-b9c4-f293e41bdf23
md"""
# Error bounds

An error of the product of two quaternions is bounded as follows (_Joldes, M.; Muller, J. M., 
Algorithms for manipulating quaternions in floating-point arithmetic.
In IEEE 27th Symposium on Computer Arithmetic (ARITH), Portland, OR, USA, 2020, pp. 48-55_)

$$
| fl(p q) - p q | \leq  (5.75\varepsilon  + \varepsilon^2) |p| |q|.$$

This implies bound error bound for dot product and matrix product in a usual manner. 

Combining it all together, we have the following result:
let $(\tilde \mu,\tilde x)$ be the computed eigenpair of the matrix $A$, where $\tilde \mu$ is in the standard form and $\|\tilde x\|_2=1$. 
Then

$$
\min_{\lambda_i \in \Lambda_s(A)} \{ | \lambda_i  - \tilde \mu | \} \leq \kappa (X) \|r\|_2.$$

"""

# ╔═╡ 4c7dba08-b8a3-4404-be88-05649e83e57f
md"
# Methods for matrices of reduced biquaternions

Many algorithms can be derived from splittings: let $A=A_1e_1+A_2e_2$. Let 

$$
A_1=Q_1 T_1Q_1^*,\quad A_2=Q_2 T_2Q_2^*$$

be the respective __complex__ Schur factorizations (which always exist). Then

$$
A=(Q_1e_1+Q_2e_2)(T_1e_1+T_2e_2)(Q_1^*e_1+Q_2^*e_2)$$

is the Schur factorization of $A$. (The proof is easy)
"

# ╔═╡ ec31e5a2-3f9b-4c37-8e9c-814322844eb4
md"
# Problem and remedy

### Problem
* QR factorization can only be used without pivoting (since the pivoting for $A_1$ and $A_2$ might be different)
* Gaussian elimination __cannot__ be used since it can run into a non-invertible pivot element.

### Remedy

Compute functions `reflector!()` and `givensAlgorithm()` using splittings.

The rest works!
```jldoctest
function LinearAlgebra.givensAlgorithm(f::T, g::T) where T<:RBiQuaternion
    v=[f;g]
    s=splitc(v)
    g1,r1=LinearAlgebra.givens(s.c1[1],s.c1[2],1,2)
    g2,r2=LinearAlgebra.givens(s.c2[1],s.c2[2],1,2)
    g1.c*e₁+g2.c*e₂, g1.s*e₁+g2.s*e₂,r1*e₁+r2*e₂
end
```
"

# ╔═╡ 519e93ec-9b60-4340-a205-2f69cbde93c1
md"
# Singular value decomposition

Having adequate functions `reflector!()` and `givensAlgorithm()`, 
the (generic) functions `bidiagonalize()` and `_svd!()` from the package 
`GenericLinearAlgebra.jl` readily work.

The latter function has a very nice implementation (see `__svd!(B, U, Vᴴ, tol = tol)`)

For matrices of reduced biquaternions, one can also compute the SVD using splitting!
"

# ╔═╡ 6b8c18ce-d001-43cc-8db6-d5de3419b833
md"
# Pseudoinverse

Having SVD, the Moore-Penrose inverse is defined as usual:
```jldoctest
function LinearAlgebra.pinv(A::Matrix{T},tol::Real=1.0e-14) where T<:RBiQuaternion
	S=svd(A)
	n=length(S.S)
	Σ=pinv.(S.S)
	return S.V*Diagonal(Σ)*S.U'
end
```
"

# ╔═╡ 72bac177-9be8-4f18-a22f-e20ef3bdb739
md"""
# Inner inverse ($1$-inverse, $AXA=A$)

Inner inverse ($1$-inverse, $AXA=A$) is defined as (_Adi Ben-Israel and Thomas N.E. Greville,
Generalized Inverses - Theory and Applications, Second Edition, Springer-Verlag New York, 2003, Section 1.2_)

Assume $A\in \mathbb{F}^{n\times n}$, where $\mathbb{F}\in\{ \mathbb{R},\mathbb{C},\mathbb{Q},\mathbb{Q}_\mathbb{R}\}$, and $\operatorname{rank}(A)=r<n$.
Let $U \begin{bmatrix} \Sigma_r & 0 \\ 0 & 0 \end{bmatrix} V^*$ be the SVD of $A$.

Set $\Sigma_1=\begin{bmatrix} \Sigma_r^{-1} & 0 \\ 0 & M \end{bmatrix}$ where $M$ is non-singular. Let $E=\Sigma_1 U^*$. Then $E$ is non-singular. 

Let $V=\begin{bmatrix} V_r & V_0 \end{bmatrix}$, where $V_r$ is
$n\times r$ part of $V$. 
Set $P=\begin{bmatrix} V_r & N \end{bmatrix}$, where $N$ is a $n\times (n-r)$ matrix such that $P$ is non-singular.

$$
EAP=\begin{bmatrix} \Sigma_r^{-1} & 0 \\ 0 & M \end{bmatrix} U^* U \begin{bmatrix} \Sigma_r & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} V_r^* \\ V_0^* \end{bmatrix}\begin{bmatrix} V_r & N \end{bmatrix}
= \begin{bmatrix} I_r & K \\ 0 & 0\end{bmatrix},$$
	
where $K=V_r^*N$. 

For a rectangular matrix $X$ some of the blocks may be missing, depending on the rank. Usually, $M$ and $N$ can be chosen as random matrices of the appropriate sizes. 
"""

# ╔═╡ 130ca7b5-e704-4e6d-9614-f48e33aa123d
md"""
# Generic code for $1$-inverse
```jldoctest
function inv₁(A::AbstractMatrix{T},tol::Real=1e-12) where T
	m,n=size(A)
	S=svd(A,full=true)
	r=rank(A)
	Σ=zeros(T,m,m)
	for i=1:r
		Σ[i,i]=S.S[i]
	end
	Σ[r+1:m,r+1:m]=randn(T,m-r,m-r)
	E=pinv(Σ)*S.U'
	P=[S.V[:,1:r] randn(T,n,n-r)]
	Ir=zeros(T,n,m)
	for i=1:r 
		Ir[i,i]=one(T)
	end
	L=randn(T,n-r,m-r)
	Ir[r+1:n,r+1:m]=L
	return P*Ir*E
end
```
"""

# ╔═╡ a23a86c9-6516-4d08-a014-62d2b6c9587a
md"
# Outer inverse ($2$-inverse, $XAX=X$)

_Predrag S. Stanimirović, Miroslav Ćirić, Igor Stojanović, and Dimitrios Gerontitis,
Conditions for Existence, Representations, and Computation of
Matrix Generalized Inverses, Complexity
Volume 2017, Article ID 6429725_

The paper considers real and complex matrices, but most of the results hold for matrices of quaternions and reduced biquaternions.

The codes are generic. Different inverse in obtained by changing $B$.
```jldoctest
inv₂(A,B)=B*inv₁(A*B)
```
The inverse $X$ also satisfies $(AX)^*=AX$.

__Solving linear equation $BXAB=B$ using (nonlinear) optimization__ (`NLsolve.jl`)
```jldoctest
function inv₂nl(A::Matrix{T},B::Matrix{T}) where T
	f!(X)=reinterpret(Float64,B*reinterpret(T,X)*A*B-B)
	X₀=reinterpret(Float64,randn(T,size(B')))
	sol=nlsolve(f!,X₀)
	U=reinterpret(T,sol.zero)
	return B*U
end
```
"

# ╔═╡ ed7fb207-7a94-491e-914b-d38979d4bbac
md"""
# 1-2 inverse
See also _Neha Bhadala,  Sk. Safique Ahmad, and  Predrag S. Stanimirović, Outer inverses of reduced biquaternion matrices, in preparation_.

Different inverse in obtained by changing $C$.
```jldoctest
inv₁₂(A,C)=inv₁(C*A)*C
```
The inverse $X$ also satisfies $(XA)^*=XA$.

__Solving $CAXC=C$ using optimization__
```
function inv₁₂nl(A::Matrix{T},C::Matrix{T}) where T
	f!(X)=reinterpret(Float64,C*A*reinterpret(T,X)*C-C)
	X₀=reinterpret(Float64,randn(T,size(C')))
	sol=nlsolve(f!,X₀)
	U=reshape(reinterpret(T,sol.zero),size(C'))
	return U*C
end
```
"""

# ╔═╡ e747a6e4-70df-4aff-993a-e9a9ad51fa03
md"
# Codes and reference

The Julia codes will be available at [https://github.com/ivanslapnicar/MANAA](https://github.com/ivanslapnicar/MANAA)

Papers are being submitted.
"

# ╔═╡ 28ec0511-644d-4c85-af15-fdf42d15c69b
md"""
# Conclusions

* Most parts of basic LA algorithms can be implemented in a generic way.
* The only "non-generic" functions are the computation of Householder reflectors and Givens rotation parameters.
* Isomorphisms to $\mathbb{C}^{2\times 2}$ help in defining conjugation.
* Applications to (some) more difficult problems (like generalized inverses) are straightforward. 
* Results can be extended to other number systems (dual numbers).
"""

# ╔═╡ a1a4a919-82fa-4a39-9cdc-92ec13b45078
md"
# Thank you!
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "6e7bcec4be6e95d1f85627422d78f10c0391f199"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─3015a3a0-0469-4ffe-b909-d28ddbbd22b4
# ╟─fd4fa4bb-c34b-4557-b17a-57b8fc3888d3
# ╟─41db52a5-812e-49bb-9d1f-fb7851740a88
# ╟─0f0099d9-5183-4aea-accb-2ae4e088172d
# ╟─9bbeb6fc-3500-4fef-8e26-758b20b581b8
# ╟─215b1cbf-5c80-4229-914e-f830e9a19150
# ╟─caccc0ce-fdec-461a-89f3-4a14ba1f93eb
# ╟─373c203c-c624-45a7-a7ea-abc79f2b1a48
# ╟─8cc9e8f5-2a27-4288-af7c-8e6d53276693
# ╟─2552eeb1-c631-4e58-94f1-3894b9545ab9
# ╟─758580cc-ad78-4ce2-a427-16c014303d11
# ╟─75268c27-c93e-45af-8cec-d3c1e74f4b7c
# ╟─fa9e44d2-5729-45c1-9aa3-a2d35372aebd
# ╟─b299fcf7-7ced-45d1-a55c-74482ecb0c60
# ╟─82c8a221-3125-42a1-a033-02408688b6ae
# ╟─02afde06-04e6-44ae-b8c4-3e8e00d5b7cc
# ╟─0b1cd856-9f43-4e2f-bade-ad01df6aee0e
# ╟─ba519f07-3941-4144-b9c4-f293e41bdf23
# ╟─4c7dba08-b8a3-4404-be88-05649e83e57f
# ╟─ec31e5a2-3f9b-4c37-8e9c-814322844eb4
# ╟─519e93ec-9b60-4340-a205-2f69cbde93c1
# ╟─6b8c18ce-d001-43cc-8db6-d5de3419b833
# ╟─72bac177-9be8-4f18-a22f-e20ef3bdb739
# ╟─130ca7b5-e704-4e6d-9614-f48e33aa123d
# ╟─a23a86c9-6516-4d08-a014-62d2b6c9587a
# ╟─ed7fb207-7a94-491e-914b-d38979d4bbac
# ╟─e747a6e4-70df-4aff-993a-e9a9ad51fa03
# ╟─28ec0511-644d-4c85-af15-fdf42d15c69b
# ╟─a1a4a919-82fa-4a39-9cdc-92ec13b45078
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
