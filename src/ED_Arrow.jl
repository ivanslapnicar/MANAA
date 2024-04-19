### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ ba49a550-fe54-11ed-1d8b-9b5192874bc6
using LinearAlgebra, Random, PlutoUI, Quaternions, GenericLinearAlgebra, Plots, JLD2

# ╔═╡ 45cb51a4-9eea-4fa8-8748-8f6ba24e9c48
TableOfContents(title="Contents", aside=true)

# ╔═╡ 484df839-8669-4627-a3c3-919300c1c882
begin
	# Structures
	struct Arrow{T} <: AbstractMatrix{T}
    	D::AbstractVector{T}
    	u::AbstractVecOrMat{T}
    	v::AbstractVecOrMat{T}
    	α
    	i::Int
	end

	struct DPRk{T} <: AbstractMatrix{T}
    	Δ::AbstractVector{T}
    	x::AbstractVecOrMat{T}
    	y::AbstractVecOrMat{T}
		ρ
	end
end

# ╔═╡ 53a198ac-e7bf-4dc2-9ae3-67f94f15a694
begin
	import Base:*,-
	function *(A::Arrow,z::Vector)
	    n=size(A,1)
	    T=typeof(A.u[1])
	    w=Vector{T}(undef,n)
	    i=A.i
	    zi=z[i]
	    for j=1:i-1
	        w[j]=A.D[j]*z[j]+A.u[j]*zi
	    end
		ind=[1:i-1;i+1:n]
		w[i]=A.v⋅z[ind]+A.α*zi
	    # w[i]=adjoint(A.v[1:i-1])*z[1:i-1]+A.α*zi+adjoint(A.v[i:n-1])*z[i+1:n]
	    for j=A.i+1:n
	        w[j]=A.u[j-1]*zi+A.D[j-1]*z[j]
	    end
	    return w
	end
	
	function *(A::DPRk,z::Vector)
	    n=size(A,1)
	    T=typeof(A.x[1])
	    w=Vector{T}(undef,n)
	    β=A.ρ*(adjoint(A.y)*z)
		return Diagonal(A.Δ)*z+A.x*β
	end

	-(A::Arrow,D::Diagonal)=Arrow(A.D-D.diag[1:end-1],A.u,A.v,A.α-D.diag[end],A.i)
	-(A::DPRk,D::Diagonal)=DPRk(A.Δ-D.diag,A.x,A.y,A.ρ)

end

# ╔═╡ 23817579-826c-47fb-aeee-d67712b59ada
begin
	import Base: size, getindex
	import  LinearAlgebra: Matrix, adjoint, transpose

	# Arrowhead
	size(A::Arrow, dim::Integer) = length(A.D)+1
	size(A::Arrow)= size(A,1), size(A,1)
	
	function getindex(A::Arrow,i::Integer,j::Integer)
	    n=size(A,1)
	    if i==j<A.i; return A.D[i]
	    elseif i==j>A.i; return A.D[i-1]
	    elseif i==j==A.i; return A.α
		elseif i==A.i&&j<A.i; return adjoint(A.v[j])
		elseif i==A.i&&j>A.i; return adjoint(A.v[j-1])
		elseif j==A.i&&i<A.i; return A.u[i]
		elseif j==A.i&&i>A.i; return A.u[i-1]
	    else 
	        return zero(A.D[1])
	    # return zeros(size(A.D[i<A.i ? i : i-1],1),size(A.D[j<A.i ? j : j-1],1))
	    end
	end
	
	Matrix(A::Arrow) =[A[i,j] for i=1:size(A,1), j=1:size(A,2)]
	adjoint(A::Arrow)=Arrow(adjoint.(A.D),A.v,A.u, adjoint(A.α),A.i)
	transpose(A::Arrow)=Arrow(A.D, conj.(A.u), conj.(A.v),A.α,A.i)

	# DPRk
	size(A::DPRk, dim::Integer) = length(A.Δ)
	size(A::DPRk)= size(A,1), size(A,1)
	
	function getindex(A::DPRk,i::Integer,j::Integer)
		# This is because Julia extracts rows as column vectors
	    Aij=conj.(A.x[i,:])⋅(A.ρ*conj.(A.y[j,:]))
	    return i==j ? A.Δ[i].+Aij : Aij
	end
	
	Matrix(A::DPRk)=[A[i,j] for i=1:size(A,1), j=1:size(A,2)]
	adjoint(A::DPRk)=DPRk(adjoint.(A.Δ),A.y,A.x,adjoint(A.ρ))
	
end

# ╔═╡ 41db52a5-812e-49bb-9d1f-fb7851740a88
md"""
# Fast computations with arrowhead matrices over associative fields

#### by Ivan Slapničar, Thaniporn Chaysri and  Nevena Jakovčević Stor

#### from University of Split, FESB

This work has been fully supported by Croatian Science Foundation under the project IP-2020-02-2240 - [http://manaa.fesb.unist.hr](http://manaa.fesb.unist.hr).

$(PlutoUI.LocalResource("./HRZZ-eng-170x80-1.jpg"))

"""

# ╔═╡ 2552eeb1-c631-4e58-94f1-3894b9545ab9
md"
# Quaternions

Quaternions are a non-commutative associative number system that extends complex numbers, introduced by Hamilton ( [1853](https://openlibrary.org/books/OL23416635M/Lectures_on_quaternions), [1866](https://openlibrary.org/books/OL7211578M/Elements_of_quaternions.)). For basic quaternions $\mathbf {i}$, 
$\mathbf {j}$, and $\mathbf {k}$, the quaternions have the form

$$q=a+b\ \mathbf {i} +c\ \mathbf {j} +d\ \mathbf {k},\quad a,b,c,d, \in \mathbb{R}.$$

The multiplication table of basic quaternions is the following:

$$\begin{array}{c|ccc}
\times & \mathbf {i} & \mathbf {j} & \mathbf {k} \\ \hline
\mathbf {i} & -1 & \mathbf {k} & -\mathbf {j} \\
\mathbf {j} & -\mathbf {k} & -1 & \mathbf {i} \\
\mathbf {k} & \mathbf {j} & -\mathbf {i} & -1
\end{array}$$

Conjugation is given by 

$$\bar q=a-b\ \mathbf {i} -c\ \mathbf {j} -d\ \mathbf {k}.$$

Then, 

$$\bar q q=q\bar q=|q|^2=a^2+b^2+c^2+d^2.$$

Let $f(x)$ be a complex analytic function. The value $f(q)$, where $q\in\mathbb{H}$, is computed by evaluating the extension of $f$ to the quaternions at $q$, see ([Sudbery,1979](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/quaternionic-analysis/308CF454034EC347D4D17D1F829F8471)), for example,

$$\sqrt{q}=\pm \left(\sqrt {\frac {\|q\|+a_1}{2}} + \frac {\operatorname{imag} (q)}{\|\operatorname{imag}(q)\|} \sqrt {\frac {\|q\|-a_1}{2}}\right).$$

Basic operations with quaternions and computation of the functions of quaternions are implemented in the package [Quaternions.jl](https://github.com/JuliaGeometry/Quaternions.jl).

## Standard form

Quaternions $p$ and $q$ are __similar__ if 

$$\exists x \quad \textrm{s. t.}\quad p=x^{-1} q x.$$

This is *iff* 

$$\operatorname{real}(p)=\operatorname{real}(q)\quad \textrm{and}\quad  
\|p\|=\|q\|.$$

The __standard form__ of the quaternion $q$ is the (unique) similar quaternion $q_s$:

$$
q_s=x^{-1} qx =a + \hat b\,  \mathbf{i},\quad \|x\|=1,\quad \hat b \geq 0,$$

where $x$ is computed as follows:

> if $c=d=0$, then $x=1$,

> if $b<0$, then $x=-\mathbf{j}$, ortherwise,

> if $c^2+d^2>0$, then $x=\hat x/\|\hat x\|$, where $\hat x=\| \operatorname{imag}(q)\|+b-d\,\mathbf{j}+c\,\mathbf{k}$.

## Homomorphism 

Quaternions are homomorphic to $\mathbb{C}^{2\times 2}$:

$$
q\to \begin{bmatrix}a+b\,\mathbf{i} & c+d\, \mathbf{i}\\-c+d\, \mathbf{i} & a-b\, \mathbf{i}\end{bmatrix}\equiv C(q),$$

with eigenvalues $q_s$ and $\bar q_s$.
"

# ╔═╡ 68b2efe8-99b0-40cf-ba11-6eb8cfa0a4d9
md"
# Matrices

All matrices are in $\mathbb{F}^{n\times n}$ where $\mathbb{F}\in\{\mathbb{R},\mathbb{C},\mathbb{H}\}$.
$\mathbb{H}$ is a non-commutative field of quaternions.
  
__Arrowhead matrix__ (__Arrow__) is a matrix of the form

$$
A=\begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix},$$

where 

$$\mathop{\mathrm{diag}}(D),u, v \in\mathbb{F}^{n-1},\quad \alpha\in\mathbb{F},$$

or any symmetric permutation of such a matrix.

__Diagonal-plus-rank-one matrix__ (__DPR1__) is a matrix of the form 

$$
A=\Delta+x ρy^*$$

where 

$$\mathop{\mathrm{diag}}(\Delta),x, y \in\mathbb{F}^{n},\quad  \rho \in \mathbb{F}.$$
"

# ╔═╡ 8503c651-2bd3-43f2-9d4c-959962606ff5
md"
# Matrix × vector

Products 

$$
w=Az$$

are computed in $O(n)$ operations. 

Let $A=\operatorname{Arrow}(D,u,v,\alpha)$. Then

$$
\begin{aligned}
w_j&=d_jz_j+u_jz_i, \quad j=1,2,\cdots,i-1\\
% y_i&=\sum_{j=1}^{i-1}\bar v_j x_j+\alpha x_i +\sum_{j=i}^{n-1}\bar v_j x_{j+1}\\
w_i&=v_{1:i-1}^* z_{1:i-1} +\alpha z_i + v_{i:n-1}^* z_{i+1:n} \\
w_j&=u_{j-1}z_i+d_{j-1}z_j,\quad j=i+1,i+2,\cdots,n.
\end{aligned}$$
"

# ╔═╡ 47f842fa-063d-4b30-a734-3f7d825b1314
md"""
# Inverses

Inverses are computed in $O(n)$ operations.

Let $A=\operatorname{Arrow}(D,u,v,\alpha)$ be nonsingular.

Let $P$ be the permutation matrix of the permutation $p=(1,2,\cdots,i-1,n,i,i+1,\cdots,n-1)$. 

If all $d_j\neq 0$, the inverse of $A$ is a DPRk (DPR1) matrix

$$
A^{-1} =\Delta+x \rho y^*,$$

where 

$$
\Delta=P\begin{bmatrix}D^{-1} & 0\\ 0 & 0\end{bmatrix}P^T,
\quad x=P\begin{bmatrix}D^{-1}u \\-1\end{bmatrix}\rho,\quad
y=P\begin{bmatrix}D^{-\star}v \\-1\end{bmatrix},\quad
\rho=(\alpha-v^\star D^{-1} u)^{-1}.$$


If $d_j=0$, the inverse of $A$ is an Arrow with the tip of the arrow at position $(j,j)$ and zero at position $A_{ii}$ (the tip and the zero on the shaft change places). In particular, let $\hat P$ be the permutation matrix of the permutation $\hat p=(1,2,\cdots,j-1,n,j,j+1,\cdots,n-1)$. Partition $D$, $u$ and $v$ as

$$
D=\begin{bmatrix}D_1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & D_2\end{bmatrix},\quad
u=\begin{bmatrix} u_1 \\ u_j \\u_2\end{bmatrix},\quad 
v=\begin{bmatrix} v_1 \\ v_j \\v_2\end{bmatrix}.$$

Then 

$$
A^{-1}=P\begin{bmatrix} \hat D & \hat u\\ \hat v^* & \hat \alpha \end{bmatrix}P^T,
$$

where

$$
\begin{align*}
\hat D&=\begin{bmatrix}D_1^{-1} & 0 & 0 \\ 0 & D_2^{-1} & 0 \\ 0 & 0 & 0\end{bmatrix},\quad
\hat u= \begin{bmatrix}-D_1^{-1}u_1 \\ -D_2^{-1}u_2\\ 1 \end{bmatrix} u_j^{-1},\quad
\hat v= \begin{bmatrix}-D_1^{-\star}v_1 \\ -D_2^{-\star}v_2\\ 1\end{bmatrix}v_j^{-1},\\
\hat \alpha&=v_j^{-\star}\left(-\alpha +v_1^\star D_1^{-1} u_1+v_2^\star D_2^{-1}u_2\right) u_j^{-1}.
\end{align*}$$
"""

# ╔═╡ b299fcf7-7ced-45d1-a55c-74482ecb0c60
md"
# Eigenvalue decomposition

Right eigenpairs $(λ,x)$ satisfy

$$
Ax=xλ, \quad x\neq 0.$$

Usually, $x$ is chosen such that $\lambda$ is the standard form.

Eigenvalues are invariant under similarity.

> Eigenvalues are __NOT__ shift invariant, that is, eigenvalues of the shifted matrix are __NOT__ the shifted eigenvalues. 

If $\lambda$ is in the standard form, it is invariant under similarity with complex numbers.  
"

# ╔═╡ 82c8a221-3125-42a1-a033-02408688b6ae
md"
## A Quaternion QR algorithm

The method is described in the paper 

[BGBM89] Angelika Bunse-Gerstner, Ralph Byers, and Volker Mehrmann, A Quaternion QR Algorithm, Numer. Math 55, 83-95 (1989)

Given a matrix $A\in\mathbb{Q}^{n\times n}$, the algorithm has four steps:

1. Reduce $A$ to Hessenberg form by Householder reflectors:

$$
X^*AX=H,$$

where $X$ is unitary and $H$ is an upper Hessenberg matrix.

2. Compute the Schur decomposition of $H$,

$$
Q^*HQ=T,$$

where $Q$ is unitary and $T$ is upper triangular matrix with eigenvalues of $A$ on the diagonal, $\Lambda=\operatorname{diag}(T)$.

3. Compute the eigenvectors $V$ of $T$ by solving the Sylvester equation:

$$
TV-V\Lambda=0.$$

Then $V^{-1}TV=\Lambda$.

4. Multiply 

$$U=X*Q*V.$$

Then $U^{-1}AU=\Lambda$ is the eigenvalue decomposition of $A$.

Since the algorithm is derived for general matrices, it requires $O(n^3)$ operations. However, the algorithm is stable and we use it in section on Numerical experiments for comparison. 

The entire algorithm is found in the function `eigen()`.
"

# ╔═╡ 09fe2fc2-5ea7-428d-b0b4-f11221fbd6d3
md"
### Reduction to Hessenberg matrix

The reduction of a quaternionic matrix to Hessenberg form is computed similarly to the real or complex case. Details of the method are given in Algorithms 1-3 in the Appendix of [BGBM89].

The Julia implementation is given in the function `hessenberg()`.
"

# ╔═╡ 02afde06-04e6-44ae-b8c4-3e8e00d5b7cc
md"
###  Computing the Schur decomposition

Given the upper Hessenber matrix $A \in\mathbb{Q}^{n\times n}$, the method 
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

Details of the method are given in Algorithm 4 in the Appendix of [BGBM89].

The Julia implementation is given in the function `schur()`.
"


# ╔═╡ 9d38f85e-058a-426c-ae71-2f664e8357a2
md"
### Solving the Sylvester equation

The Sylvester equation 

$$
TV-V\operatorname{diag}T=0$$

is solved as in the real or complex case. 

The Julia implementation is given in the function `sylvester()`.
"

# ╔═╡ f6769f8f-19ad-47c5-a1ec-2e3c780f6cd7
md"
## RQI

Starting with the unit-norm vector $x_0$, the __Rayleigh Quotient Iteration__ (RQI) produces sequences of shifts and vectors

$$
\mu_k=\operatorname{real}(x_k^*Ax_k),\quad y_k=(A-\mu_kI)^{-1} x_k, \quad  x_{k+1}=\frac{y_k}{\| y_k\|},\quad k=0,1,2,\ldots$$

> Due to fast inverses, one step of the methods requires $O(n)$ operations.

Since the eigenvalues are not shift invariant, only real shifts can be used. However, this works fine due to the following: let the matrix $A -\mu I$ have purely imaginary standard eigenvalue:

$$(A-\mu I)x=x (i λ),\qquad \mu,\lambda\in\mathbb{R}.$$

Then 

$$Ax=\mu x+xi\lambda=x(\mu+i\lambda).$$

The method converges, but, since only real shifts are used, the convergence is rather slow and requires a large number of iterations. 
"

# ╔═╡ 24724b41-f108-4dc7-b6dd-193a3e39bc37
md"
## RQI with double shifts

We can apply the double shift $\mu$ and $\bar \mu$ similarly as in the [BGGM] method. 

The __Rayleigh Quotient Iteration with Double Shifts__ (RQIds) produces sequences of shifts and vectors

$$
\mu_k=\frac{1}{x_k^*x_k} x_k^*Ax_k,\quad 
y_k=(A^2-(\mu_k+\bar\mu_k)A+\mu_k\bar\mu_k I)^{-1} x_k, \quad  
x_{k+1}=\frac{y_k}{\| y_k\|},\quad k=0,1,2,\ldots$$

> Due to the arrowhead structure of $A$, one step of the method requires $O(n)$ operations:

 $y_k$ is the solution of the system

$$
(A^2-(\mu_k+\bar\mu_k)A+\mu_k\bar\mu_k I) y_k=x_k.$$

Let 

$$
y_k=y,\quad
x_k=\begin{bmatrix} x \\ \xi \end{bmatrix},\quad  \hat\alpha=\mu_k+\bar\mu_k,\quad 
\beta=\mu_k\bar\mu_k.$$

Notice that $\hat \alpha$ and $\beta$ are real. Then:

$$
\left(\begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} \begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} -\hat\alpha \begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} + \beta \begin{bmatrix} I & 0 \\ 0  & 1 \end{bmatrix} \right) y = \begin{bmatrix} x \\ \chi \end{bmatrix}.$$
"

# ╔═╡ a341e994-50a7-4848-8ec8-8794a89a9063
md"
Therefore,

$$
\left(\begin{bmatrix} D^2 + uv^* & Du +u\alpha \\ v^*D+\alpha v^* & v^*u+\alpha^2 \end{bmatrix} -\hat\alpha \begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} + \beta \begin{bmatrix} I & 0 \\ 0  & 1 \end{bmatrix} \right) y = \begin{bmatrix} x \\ \chi \end{bmatrix},$$

so

$$
\begin{bmatrix} D^2 -\hat\alpha D +\beta I +uv^* & 
Du +u(\alpha-\hat\alpha) \\ 
v^*D+(\alpha-\hat\alpha) v^* & 
v^*u+(\alpha-\hat\alpha)\alpha+\beta \end{bmatrix}y = \begin{bmatrix} x \\ \chi \end{bmatrix}. \tag{1}$$
"

# ╔═╡ d3046577-b251-45b0-a743-a9970937811d
md"
The matrix $C=D^2 -\hat\alpha D +\beta I +uv^*$ is a DPRk (DPR1) matrix

$$
C=\operatorname{DPRk}(D^2 -\hat\alpha D +\beta I,u,v,1).$$

Multiplication of equation (1) by the block matrix $\begin{bmatrix} C^{-1} & \\ & 1\end{bmatrix}$ from the left yields

$$
\begin{bmatrix} I & 
C^{-1}(Du +u(\alpha-\hat\alpha)) \\ 
v^*D+(\alpha-\hat\alpha) v^* & 
v^*u+(\alpha-\hat\alpha)\alpha+\beta \end{bmatrix}y = \begin{bmatrix} C^{-1}x \\ \xi \end{bmatrix}. \tag{2}$$

Let $M$ denote the matrix on the left hand side of equation (2). Then $M$ is an arrowhead matrix and, finally, $y=M^{-1}z$, where $z=\begin{bmatrix} C^{-1}x \\ \xi \end{bmatrix}$.

Due to fast computation of inverses of arrowhead and DPRk matrices from Lemmas 2 and 3, respectively, one step of the RQIds method requires $O(n)$ operations. 
"

# ╔═╡ 22c35821-40f4-4c64-90b3-2ea2ce4e651c
md"
## Wielandt's deflation

* Let $A$ be a (real, complex, or quaternionic) matrix.
* Let $(\lambda,u)$ be a right eigenpair of $A$.
* Choose $z$ such that $z^* u=1$, say $z^*=\begin{bmatrix} 1/u_1 & 0 & \cdots & 0\end{bmatrix}$.
* Compute the deflated matrix $\tilde A=(I-uz^*)A$.
* Then  $(0,u)$ is an eigenpair of $\tilde A$.  
* Further, if $(\mu,v)$ is an eigenpair of $A$, then $(\mu, \tilde v)$, where $\tilde v=(I-uz^*)v$ is an eigenpair of $\tilde A$.

__Proofs:__ Using $Au=u\lambda$ and $z^*u=1$, the first statement holds since 

$$
\tilde A u= (I-uz^*)A u=Au-uz^*Au=u\lambda-uz^*u\lambda=0.$$

Further,

$$
\begin{aligned}
\tilde A \tilde v &=(I-uz^*)A(I-uz^*)v \\
&= (I-uz^*)Av-Auz^*v+uz^*Auz^*v \\
&= (I-uz^*)v\mu -u\lambda z^*v+uz^*u\lambda z^* v \\
&= \tilde v\mu
\end{aligned}.$$
"

# ╔═╡ fa8ead94-9787-462b-9f41-47fcb41a1a17
md"
## Arrowhead matrices

__Lemma 1.__ Let $A$ be an arrowhead matrix partitioned as

$$
A=\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix},$$

where $\chi$, $\upsilon$ and $\alpha$ are scalars, $x$ and $y$ are vectors, and $\Delta$ is a diagonal matrix. 

Let $\bigg(\lambda,\begin{bmatrix}\nu \\ u \\ \psi \end{bmatrix}\bigg)$, where $\nu$ and $\psi$ are scalars, and $u$ is a vector, be an eigenpair of $A$. Then, the deflated matrix $\tilde A$ has the form 

$$
\tilde A=\begin{bmatrix} 0 & 0^T \\ w & \hat A\end{bmatrix}, \tag{1}$$

where 

$$
w=\begin{bmatrix}-u\frac{1}{\nu}\delta \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon}\end{bmatrix},$$

and $\hat A$ is an arrowhead matrix

$$
\hat A=\begin{bmatrix} \Delta  & -u\frac{1}{\nu}\chi
+x \\ y^*
& -\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix}.\tag{2}$$

__Proof:__ We have

$$
\begin{aligned}
\tilde A&=\bigg(\begin{bmatrix} 1 & 0^T & 0 \\ 0 & I & 0 \\ 0 & 0^T & 1 \end{bmatrix} - 
\begin{bmatrix} \nu \\ u \\ \psi \end{bmatrix}  \begin{bmatrix} \frac{1}{\nu} & 0^T & 0 \end{bmatrix} \bigg)
\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}
\\
&= \begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu} & I & 
0 \\ -\psi \frac{1}{\nu} & 0^T & 1 \end{bmatrix}
\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}\\
&= \begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu}\delta & \Delta & 
-u \frac{1}{\nu}\chi+x \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon} & y^* & 
-\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix},
\end{aligned} \tag{3}$$

as desired. $\square$
"

# ╔═╡ ff113c87-a72d-4556-98f9-e1e42782a1e6
md"

__Lemma 2.__ Let $A$ and $\hat A$ be as in Lemma 4. If $\bigg(\mu, \begin{bmatrix}\hat z \\ \hat \xi\end{bmatrix}\bigg)$ is an eigenpair of $\hat A$, then the eigenpair of $A$ is

$$
\left(\mu, \displaystyle\begin{bmatrix}\zeta \\ \hat z + u\frac{1}{\nu}\zeta \\ \hat \xi + \psi\frac{1}{\nu}\zeta\end{bmatrix}\right), \tag{4}$$

where $\zeta$ is the solution of the scalar Sylvester equation 

$$\bigg(\delta+\chi\psi\frac{1}{\nu}\bigg)\zeta-\zeta \mu=-\chi\hat \xi. \tag{5}$$


__Proof:__ If $\mu$ is an eigenvalue of $\hat A$, it is obviously also an eigenvalue of $\tilde A$, and then also of $A$. Assume that the corresponding eigenvector of $A$ is partitioned as $\displaystyle\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}$. By combining (1), (2) and (3), we have 

$$
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu}\delta & \Delta & 
-u \frac{1}{\nu}\chi+x \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon} & y^* & 
-\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix}
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu} & I & 
0 \\ -\psi \frac{1}{\nu} & 0^T & 1 \end{bmatrix}
\displaystyle\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}=
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu} & I & 
0 \\ -\psi \frac{1}{\nu} & 0^T & 1 \end{bmatrix}
\displaystyle\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}\mu,$$

or

$$
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu}\delta & \Delta & 
-u \frac{1}{\nu}\chi+x \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon} & y^* & 
-\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix}
\begin{bmatrix} 0  \\ -u\frac{1}{\nu}\zeta+z \\ -\psi \frac{1}{\nu}\zeta + \xi\end{bmatrix}=
\begin{bmatrix} 0  \\ -u\frac{1}{\nu}\zeta+z \\ -\psi \frac{1}{\nu}\zeta + \xi\end{bmatrix}\mu,$$

Since the bottom right $2\times 2$ matrix is $\hat A$, the above equation implies

$$
\begin{bmatrix}\hat z \\ \hat \xi\end{bmatrix}=\begin{bmatrix} -u\frac{1}{\nu}\zeta+z \\ -\psi \frac{1}{\nu}\zeta + \xi\end{bmatrix},$$

which proves (4). It remains to compute $\zeta$. The first component of the equality 

$$
A\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}=\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}=\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}\mu$$

implies

$$
\delta \zeta +\chi \xi=\zeta \mu,$$

or 

$$\delta \zeta + \chi \big(\hat \xi +\psi \frac{1}{\nu}\zeta\big)=\zeta \mu,$$ 

which is exactly the equation (5). $\square$
"

# ╔═╡ 88e05838-d19d-45b8-b7ad-ca1fb6d47f7b
md"
### Computing the eigenvectors

Let $\left(\lambda,\begin{bmatrix} \nu \\ u \\ \psi\end{bmatrix}\right)$ be an eigenpair of the matrix $A$, that is

$$
\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}\begin{bmatrix}\nu \\ u \\ \psi\end{bmatrix}=\begin{bmatrix}\nu \\ u \\ \psi\end{bmatrix}\lambda.$$


If $\lambda$ and $\psi$ are known, then the other components of the eigenvector are solutions of scalar Sylvester equations

$$
\begin{aligned}
\delta \nu -\nu \lambda & = -\chi \psi,\qquad\qquad\qquad\qquad (6)\\
\Delta_{ii}u_i-u_i\lambda & =-x_i\psi,\quad i=1,\ldots,n-2.
\end{aligned}$$

By setting

$$
\gamma=\delta+\chi\psi\frac{1}{\nu}$$

the Sylvester equation (5) becomes

$$
\gamma\zeta-\zeta \mu=-\chi\hat \xi. \tag{7}$$

Dividing (16) by $\nu$ from the right gives

$$\gamma=\nu\lambda\frac{1}{\nu}. \tag{8}$$
"

# ╔═╡ a859ec97-d40a-4d35-908d-ccdc16c5fd57
md"
### Algorithm

In the first (forward) pass, in each step the absolutely largest eigenvalue and its eigenvector are computed by the power method, RQI or MRQI. The first element of the current vector $x$ and the the first and the last elements of the current eigenvector are stored. The current value $\gamma$ is computed using (8) and stored. The deflation is then performed according to Lemma 4.

The eigenvectors are reconstructed bottom-up, that is from the smallest matrix to the original one (a backward pass). In each iteration we need to have the access to the first element of the vector $x$ which was used to define the current Arrow matrix, its absolutely largest eigenvalue, and the first and the last elements of the corresponding eigenvector.

In the $i$th step, for each $j=i+1,\ldots, n$ the following steps are performed:

1. The equation (7) is solved for $\zeta$ (the first element of the eigenvector of the larger matrix). The quantity $\hat \xi$ is the last element of the eigenvectors and was stored in the forward pass. 
2. The first element of eigenvector of super-matrix is updated (set to $\zeta$).
3. The last element of the eigenvectors of the super matrix is updated using (4). 

Iterations are completed in $O(n^2)$ operations.  

After all iterations are completed, we have:

* the absolutely largest eigenvalue and its eigenvector (unchanged from the first run of the Power Method), 
* all other eigenvalues and the last elements of their corresponding eigenvectors.  

The rest of the elements of the remaining eigenvectors are computed using the procedure described at the beginning of the previous section. This step also requires $O(n^2)$ operations.

Due to floating-point error in operations with Quaternions, the computed eigenpairs have larger residuals that required. This is successfully remedied by running again few steps of the RQIds, but starting from the computed eigenvectors. This has the effect of using nearly perfect shifts, so typically just a few additional iterations are needed to attain the desired accuracy. This step also requires $O(n^2)$ operations.

"

# ╔═╡ e747a6e4-70df-4aff-993a-e9a9ad51fa03
md"
# Code
"

# ╔═╡ cf8174d9-035d-4463-ba8b-88b1b6b44317
md"
## Quaternions
"

# ╔═╡ 3f8094c4-93ce-4b01-9b50-c7d66031a610
begin
	import Base: eps, imag
	eps(::Type{Complex{T}}) where T=eps(T)
	eps(::Type{Quaternions.Quaternion{T}}) where T=eps(T)
	const QuaternionF64=Quaternion{Float64}
	im=Quaternion(0,1,0,0)
	jm=Quaternion(0,0,1,0)
	km=Quaternion(0,0,0,1)
	Quaternion{T}(x::Complex) where {T<:Real} = Quaternion(convert(T,real(x)),convert(T,imag(x)),0,0)
	imag(q::Quaternion)=(q-conj(q))/2

	# Quaternion to 2x2 complex
	function q2c(c::T) where T<:Number
	    return [complex(c.s, c.v1) complex(c.v2,c.v3);
	        	complex(-c.v2,c.v3) complex(c.s,-c.v1)]
	end

	function c2q(A::Matrix)
		q=Quaternion(real(A[1,1]),imag(A[1,1]),real(A[1,2]),imag(A[1,2]))
	end
	
	# For compatibility
	function q2c(c::T) where T
	    return c
	end

	 # Converts matrix to matrix of 2x2 blocks 
	function block(A::Array{T},k::Int) where T
    	# Partition matrix into equally sized blocks
    	m,n=div(size(A,1),k),div(size(A,2),2)
    	B=[zeros(T,k,k) for i=1:m,j=1:n]
    	for i=1:m
        	for j=1:n
            	B[i,j]=A[k*(i-1)+1:k*i,k*(j-1)+1:k*j]
        	end
    	end
    	B
	end
	
	# Converts block matrix to ordinary matrix
	unblock(A) = mapreduce(identity, hcat, [mapreduce(identity, vcat, A[:,i]) 
        for i = 1:size(A,2)])
end

# ╔═╡ 9370d0a1-2ae4-4f8b-8da9-45339eeb21b4
begin
	# Test the arithmetic
	p=randn(QuaternionF64)
	q=√p
	abs(q*q-p)
end

# ╔═╡ ddd9a82c-5d26-42be-80e1-087ba826ee10
md"
###  Standard form
"

# ╔═╡ dc11a4fe-c905-4cb5-a90d-8256cb469a39
md"
## Matrices
"

# ╔═╡ bdeab9ab-4838-4332-9b74-7bba33ccb317
md"
## `*()`
"

# ╔═╡ 322273eb-ca50-42b1-866a-3977700e9b63
md"
## `inv()`
"

# ╔═╡ 2c23d735-c208-4a4c-b358-cf1a12a38ebe
begin
	import LinearAlgebra.inv
	function inv(A::Arrow)
	    j=findfirst(iszero.(A.D))
	    if j==nothing
	        p=[1:A.i-1;length(A.D)+1;A.i:length(A.D)]
	        Δ=inv.(A.D)
	        x=Δ.* A.u
			push!(x,-one(x[1]))
			y=adjoint.(Δ) .* A.v
			push!(y,-one(y[1]))
	        ρ=inv(A.α-adjoint(A.v)*(Δ .*A.u))
			push!(Δ,zero(Δ[1]))
	        return DPRk(Δ[p],x[p],y[p],ρ)
	    else
			n=length(A.D)
	        ind=[1:j-1;j+1:n]
	        D=A.D[ind]
	        u=A.u[ind]
	        v=A.v[ind]
	        pₕ=collect(1:n)
	        deleteat!(pₕ,n)
	        iₕ= (j>=A.i) ? A.i : A.i-1
	        insert!(pₕ,iₕ,n)

			# Little bit elaborate to acommodate blocks
			Dₕ=inv.(D)
			uₕ=-Dₕ .* u
			push!(uₕ,one(uₕ[1]))
			uₕ*=inv(A.u[j])
			
	        vₕ=-adjoint.(Dₕ) .* v
			push!(vₕ,one(D[1]))
			vₕ*=inv(A.v[j])
			
	        αₕ=adjoint(inv(A.v[j]))*(-A.α+adjoint(v)*(Dₕ .* u))*inv(A.u[j])
	        
			push!(Dₕ,zero(D[1]))
			jₕ=(j<A.i) ? j : j+1
	        return Arrow(Dₕ[pₕ],uₕ[pₕ],vₕ[pₕ],αₕ,jₕ)
	    end
	end
end

# ╔═╡ 124aea10-888e-438c-b0e0-82cdc1cf6dcb
md"
## `\()`
"

# ╔═╡ 77352b2b-3642-41fe-bc8a-7046b054608b
begin 
	function Base.:\(A::Arrow{T},μ::AbstractFloat,x::Vector{T}) where T<:Number
		# Solution of a shifted system (A-μI)y=x with real shift μ
		y=similar(x)
		z=(A.D .-μ).\x[1:end-1]
		y[end]=(A.α-μ-A.v⋅((A.D .-μ).\A.u))\(x[end]-A.v⋅z)
		y[1:end-1]=z-(A.D .-μ).\(A.u*y[end])
		return y
	end

	function Base.:\(A::Arrow{T},x::Vector{T}) where T<:Number
		# Solution of a system Ay=x
		y=similar(x)
		z=A.D.\x[1:end-1]
		y[end]=(A.α-A.v⋅(A.D.\A.u))\(x[end]-A.v⋅z)
		y[1:end-1]=z-A.D.\(A.u*y[end])
		return y
	end
end

# ╔═╡ 3566ab10-c23f-4f82-a4fb-2e3a134868a3
md"
## A Quaternion QR algorithm
"

# ╔═╡ 1c0a18a8-8490-4790-b3b7-7f56edab3d43
md"
## `Power()`
"

# ╔═╡ 488877d8-1dd7-43a0-97e1-ce12c2555f5d
md"
## `RQI()`
"

# ╔═╡ 12178ddc-c0bc-45cd-8c45-299b3ff46029
md"
## `RQIds()`
"

# ╔═╡ b909e897-bb2f-4171-9e43-caa326bae66d
function Base.:\(A::DPRk{T},x::Vector{T}) where T<:Quaternion
	ρ=-A.ρ/(one(T)+(A.y⋅(A.Δ .\A.x)*A.ρ))
	A.Δ .\ (x+A.x.*(ρ*(A.y⋅(A.Δ.\x))))
end

# ╔═╡ b2d0628a-35c8-4d8e-ba25-480173557229
function Base.:\(A::Arrow{T},α::Number, β::Number, x::Vector{T}) where T<:Quaternion
	# Solution of a double shifted system (A^2-αA+βI)y=x with a Quaternion
	# shift μ, where α=μ+conj(μ) and β=μ*conj(μ)
	n=length(x)
	B=DPRk(A.D.*(A.D.-α).+β,A.u,A.v,one(T))
	# C=inv(DPRk(A.D.^2-α*A.D.+β,A.u,A.v,one(T)))
	u=\(B,A.D.*A.u+A.u*(A.α-α))
	z=\(B,x[1:end-1])
	v=conj.(A.D).*A.v+A.v*(conj(A.α)-α)
	α₀=A.v⋅A.u+(A.α-α)*A.α+β
	# M=Arrow(ones(T,n-1),u,v,α₀,n)
	# Solution of the system My=[z;x[end]]
	y=similar(x)
	y[end]=(α₀-v⋅u)\(x[end]-v⋅z)
	y[1:end-1]=z-(u*y[end])
	return y
end

# ╔═╡ 4e9666dc-26f7-429e-afc5-c72f14a34e9a
begin
	function standardformx(a::Vector{T}) where T<:Quaternion
		# Computes vector x such that inv.(x).*a .*x is in the standard form
 		n=length(a)
		x=Array{T}(undef,n)
    	for i=1:n
        	x[i]=standardformx(a[i])
    	end
    	x
	end  

	function standardformx(a::Quaternion)
		# Return standard form of a
        b=copy(a)
        if norm([b.v2 b.v3])>0.0
            # x=BLAS.nrm2([b.v1 b.v2 b.v3])+b.v1-b.v3*jm+b.v2*km
			x=norm([b.v1 b.v2 b.v3])+b.v1-b.v3*jm+b.v2*km
            x/=norm(x) # BLAS.nrm2([x.s x.v1 x.v2 x.v3])
        elseif b.v1<0.0
            x=-jm
        else
            x=1.0
        end
        return x
	end

	function standardform(a::Quaternion)
		# Return standard form of a
        x=standardformx(a)
        # watch out for the correct division: / and not \
    	return (x\a)*x
	end

	standardform(a::Float64)=a
	standardformx(a::Float64)=one(a)
	standardform(a::ComplexF64)=a
	standardformx(a::ComplexF64)=one(a)
	standardform(a::Complex{BigFloat})=a
	standardformx(a::Complex{BigFloat})=one(a)
	
end

# ╔═╡ 6f71be95-f2bc-4f47-bbca-89bf8cd53cab
begin
	# This file containf Julia implementation of algorithms from the article
	# [BGBM89] Angelika Bunse-Gerstner, Ralph Byers, and Volker Mehrmann, A
	# Quaternion QR Algorithm, Numer. Math 55, 83-95 (1989)
	
	function HouseholderVector(x::AbstractVector{T}) where T<:Quaternion
	    # Computes the Householder vector for a Quaternion vector x
		# This is implementation of Algorithm A1 from [Appendix, BGBM89]
	    v=copy(x)
		if imag(v[1])!=zero(T)
			v/=v[1]
		end
		v[1]+=norm(v)
	    return v
	end
	
	function LinearAlgebra.hessenberg(A₁::AbstractMatrix{T}) where T<:Quaternion
		# Computes the reduction of the matrix A₁ to Hessenberg form, Q'*A₁*Q=H
		# This is implementation of Algorithm A3 from [Appendix, BGBM89]
		A=copy(A₁)
		m=size(A,1)
	    Q=Matrix{T}(I,m,m)
	    for k=1:m-1
	        v=HouseholderVector(A[k+1:m,k])
	        β=(2/(v⋅v))*v
	        A[k+1:m,k:m]-=β*(v'*A[k+1:m,k:m])
			A[:,k+1:m]-=(A[:,k+1:m]*v)*β'
	        Q[:,k+1:m]-=(Q[:,k+1:m]*v)*β'
	    end
	    return Q,triu(A,-1)
	end
	
	function schur2(A₁::AbstractMatrix{T}) where T<:Quaternion
		# Schur decomposition for 2×2 matrix of Quaternions
		A=copy(A₁)
		n=size(A,1)
		if n!=2
			println(" Wrong dimension ")
			return
		end
		Q=Matrix{T}(I,n,n)
		tol=1.0e-14
		step=0
		while abs(A[2,1])>tol*√(abs(A[1,1])*abs(A[2,2])) && step<10
			# Choose μ
			μ=A[n,n]
			# First column of M,  
			# First step of implicit method
			x₁=A[1,1]^2+A[1,2]*A[2,1]-2*real(μ)*A[1,1]+abs(μ)^2
			x₂=A[2,1]*A[1,1]+A[2,2]*A[2,1]-2*real(μ)*A[2,1]
			x=[x₁,x₂]
			# First transformation
			v=HouseholderVector(x)
	    	β=(2/(v⋅v))*v
			A-=β*(v'*A)
			A-=(A*v)*β'
	    	# Q[1:3,:]-=β*(v'*Q[1:3,:])
			Q-=(Q*v)*β'
			step+=1
		end
		# println(" Schur2 ",step)
		return triu(A),Q
	end
	
	function LinearAlgebra.schur(A₀::AbstractMatrix{T},tol::Real=1e-12) where T<:Quaternion
		# Schur factorization of an upper Hessenberg matrix A₀ of Quaternions,
		# Q'*A₀*Q=T. This is implementation of Algorithm A4 from [Appendix, BGBM89]
		A=copy(A₀)
		n=size(A,1)
		# println(" n= ",n)
		Q=Matrix{T}(I,n,n)
		if n==1
			return A, Q
		end
		if n==2
			R₀,Q₀=schur2(A)
			return R₀,Q₀
		end
		maxsteps=200
		steps=0
		k=0
		dh=zeros(T,n)
		da=zeros(T,n)
		while k==0 && steps<=maxsteps
			# Choose μ
			# Standard shift
			μ=A[n,n]
			# Wilkinson's shift
			#=
			R₀,Q₀=schur2(A[n-1:n,n-1:n])
			μ₀=diag(R₀)
			μ=μ₀[findmin(abs,μ₀.-A[n,n])[2]]
			=#
			# First column of M,  
			# First step of implicit method
			x₁=A[1,1]^2+A[1,2]*A[2,1]-2*real(μ)*A[1,1]+abs(μ)^2
			x₂=A[2,1]*A[1,1]+A[2,2]*A[2,1]-2*real(μ)*A[2,1]
			x₃=A[3,2]*A[2,1]
			x=[x₁,x₂,x₃]
			# First transformation
			v=HouseholderVector(x)
	    	β=(2/(v⋅v))*v
			A[1:3,:]-=β*(v'*A[1:3,:])
			k₁=min(n,4)
			A[1:k₁,1:3]-=(A[1:k₁,1:3]*v)*β'
	    	# Q[1:3,:]-=β*(v'*Q[1:3,:])
			Q[:,1:3]-=(Q[:,1:3]*v)*β'
			# Iterations - chasing the bulge
			for i=1:n-2
				k=min(i+3,n)
				x=A[i+1:k,i]
				v=HouseholderVector(x)
				β=(2/(v⋅v))*v
				A[i+1:k,:]-=β*(v'*A[i+1:k,:])
				A[:,i+1:k]-=(A[:,i+1:k]*v)*β'
	    		# Q[i+1:k,:]-=β*(v'*Q[i+1:k,:])
				Q[:,i+1:k]-=(Q[:,i+1:k]*v)*β'
			end
			steps+=1
	        # Deflation criterion
			dh=abs.(diag(A,-1))
			da=abs.(diag(A))
	        k=findfirst(dh .< sqrt.(da[1:n-1].*da[2:n])*tol)
	        k=k==nothing ? 0 : k
			if k>0
				# println(" k= ", k, " dh₁= ",dh[k]/√(da[k]*da[k+1]))
			end
		end
		if steps==maxsteps+1
			# No proper convergence, but move on with the relatively 
			# smallest element in the sub-diagonal)
			println(" No convergence ")
			dh₁,k=findmin(dh ./ sqrt.(da[1:n-1].*da[2:n]))
			println(" k= ",k," dh₁= ", dh₁)
		end
		# println(" steps= ", steps)
		# Split the matrices
		A₁=A[1:k,1:k]
		A₂=A[k+1:n,k+1:n]
		A₁₂=A[1:k,k+1:n]
		# Recursive call
		R₁,Q₁=schur(A₁)
		R₂,Q₂=schur(A₂)
		# Put the solution back
		A[1:k,1:k]=R₁
		A[k+1:n,k+1:n]=R₂
		A[1:k,k+1:n]=Q₁'*A[1:k,k+1:n]*Q₂
		Q[:,1:k]*=Q₁
		Q[:,k+1:n]*=Q₂
		# Return
		return triu(A),Q
	end
	
	function Quaternions.sylvester(A::AbstractMatrix{T}) where T<:Quaternion
	    n=size(A,1)
	    X=Matrix{T}(I,n,n)
	    for k=2:n
	        for i=k-1:-1:1
	            r=(conj(A[i,i+1:k])⋅X[i+1:k,k])
	            a=-A[i,i]
	            b=A[k,k]
	            X[i,k]=sylvester(a,b,-r)
	        end
	    end
	    X
	end
	
	function LinearAlgebra.eigen(A₀::AbstractMatrix{T}, standardform::Bool=true, tol::Real=1e-12) where T<:Quaternion
		# Eigenvalue decomposition of the matrix A₀ of Quaternions, Q'*A₀*Q=Λ.
		#  This is implementation of Algorithm A5 from [Appendix, BGBM89]
		A=copy(A₀)
		# Reduction to Hessenberg form
		Q,H=hessenberg(A)
		# Schur factorization
		R,Q₁=schur(H,tol)
		Q*=Q₁
		# Computing eigenvectors by solving Sylvester equation
		X=sylvester(R)
		λ=diag(R)
		Q*=X
		# Eigenvalues in the standard form
		if standardform
			for i=1:length(λ)
				z=standardformx(λ[i])
		    	λ[i]=z\(λ[i]*z)
				Q[:,i]*=z
			end
		end
		return Eigen(λ,Q)
	end
end

# ╔═╡ 15b01358-f35b-4d43-b953-d0f046760db6
function Power(A::AbstractMatrix{T},standardform::Bool=true,tol::Real=1e-12) where T<:Number
	# Right eigenvalue and eigenvector of a (quaternion) Arrow matrix
	x=normalize!(randn(T,size(A,1)))
	y=A*x
    ν=x⋅y
    steps=1
	# println(tol)
    while norm(y-x*ν)/norm(y)>tol && steps<2000
		normalize!(y)
		x=y
        y=A*x
        ν=x⋅y
        # println(ν)
        steps+=1
    end
	normalize!(y)
	if standardform
		z=standardformx(ν)
    	ν=z\(ν*z)
		if T∈(QuaternionF64,Quaternion{BigFloat})
			ν=Quaternion(ν.s,ν.v1,0.0,0.0)
		end
		y.*=z
	end
	println("Power ", steps)
    ν, y, steps
end

# ╔═╡ 7570c157-1f63-47e1-9d31-c2a690e5f55b
function RQI(A::AbstractMatrix{T},standardform::Bool=true,tol::Real=1e-12) where T<:Number
	# Right eigenvalue and eigenvector of a (quaternion) Arrow matrix 
	# using Rayleigh Quotient Iteration
	# Solves the system instead of computing the inverse
	n=size(A,1)
	x=normalize!(ones(T,n))
	# Only real shifts
    ν=x⋅(A*x)
	μ=real(ν)
	y=\(A,μ,x)
	normalize!(y)
    steps=1
    while norm(A*y-y*ν)/abs(ν)>tol && steps<4000
		x=y
		ν=x⋅(A*x)
		μ=real(ν)
        y=\(A,μ,x)
		normalize!(y)
        steps+=1
		# println(norm(A*y-y*ν)/abs(ν))
    end
	if standardform
		z=standardformx(ν)
    	ν=z\(ν*z)
		if T∈(QuaternionF64,Quaternion{BigFloat})
			ν=Quaternion(ν.s,ν.v1,0.0,0.0)
		end
		y.*=z
	end
	println("RQI ",steps)
    return ν, y, steps
end

# ╔═╡ 86221eaf-9c7e-4171-b29e-66496a1a55d1
function RQIds(A::AbstractMatrix{T}, standardform::Bool=true, tol::Real=1e-12) where T<:Number
	# Right eigenvalue and eigenvector of a (quaternion) Arrow matrix 
	# using Rayleigh Quotient Iteration and double-shift approach
	# Solves the system instead of computing the inverse
	n=size(A,1)
	x=normalize!(ones(T,n))
	# Start
	μ=x⋅(A*x)
	α=T
	β=T
    steps=0
	residual=norm(A*x-x*μ)/abs(μ)
    while residual>tol && steps<100
		# Double shifts
		α=μ+conj(μ)
		β=μ*conj(μ)
		# Direct implementation of double shift is O(n^3).
		# y=(A*A-α*A+β*I)\x
		# This is O(n^2)
		x=\(A,α,β,x)
		normalize!(x)
		# println("step = ",steps, " μ = ", μ)
		μ=x⋅(A*x)
		residual=norm(A*x-x*μ)/abs(μ)
		# println("residual = ", norm(A*x-x*μ)/abs(μ) )
        steps+=1
    end
	if standardform
		z=standardformx(μ)
    	μ=z\(μ*z)
		x.*=z
	end
	# println("RQIds steps= ",steps," residual = ", residual)
    return μ, x, steps
end

# ╔═╡ 6aa6a509-b3b3-44b4-bcad-5b0b332f3ac2
function RQIds(A::AbstractMatrix{T},μ::T, x::Vector{T},standardform::Bool=true,tol::Real=1e-12) where T<:Quaternion
	# Right eigenvalue and eigenvector of a (quaternion) Arrow matrix 
	# using Rayleigh Quotient Iteration and double-shift approach
	# Solves the system instead of computing the inverse.
	# Uses nearly optimal shift and optimal starting vector from previous 
	# computations. It is used for correction.
	n=size(A,1)
	# Start
	α=T
	β=T
    steps=0
	residual=norm(A*x-x*μ)/abs(μ)
    while residual>tol && steps<20
		α=μ+conj(μ)
		β=μ*conj(μ)
		x=\(A,α,β,x)
		normalize!(x)
		# println("step = ",steps, " μ = ", μ)
		μ=x⋅(A*x)
		residual=norm(A*x-x*μ)/abs(μ)
        steps+=1
    end
	if standardform
		z=standardformx(μ)
    	μ=z\(μ*z)
		x.*=z
	end
	# println("RQIds! steps= ",steps," residual= ", residual)
    return μ, x
end

# ╔═╡ ce762b41-6522-46d1-a332-eca6756d9687
md"
## `eigvals()`
"

# ╔═╡ 98d415a5-1cdd-48ae-bad2-46230a7df2b9
md"
## `eigvecs()`
"

# ╔═╡ 8d76c56a-60e8-4356-a0e5-3c41d01bc530
function LinearAlgebra.eigvecs(A::Arrow{T}, λ₁::Vector{T}, ψ₁::Vector{T}) where T<:Number
	# Eigenvectors of a (quaternionic) Arrow given eigenvalues λ₁ and last
	# elements ψ₁
	n=length(λ₁)
	# Create matrix for eigenvectors
	U=Matrix{T}(undef,n,n)
	# Temporary vector
	u=Vector{T}(undef,n)
	for i=1:n
		# Compute α=ρ*y'*u from the first element
		ψ=ψ₁[i]
		λ=λ₁[i]
		u[n]=ψ
		for k=1:n-1
			u[k]=sylvester(A.D[k],-λ,A.u[k]*ψ)
		end
		normalize!(u)
		U[:,i]=u
	end
	return U
end

# ╔═╡ 7e6dcca5-e545-4822-8e49-634514fd60bb
md"
## `eigen()`
"

# ╔═╡ aefbdfff-3cb2-41a4-89be-67915bfb240b
md"
## Examples
"

# ╔═╡ 8d04d6ae-17bf-447b-87e3-f7702fc8007c
begin
	T=QuaternionF64
	n=100
	Esolver=RQIds
	tol=1e-12
	matrixtype="Arrow"
	numberofexperiments=10
end

# ╔═╡ 1319217e-fa81-4188-992f-95f5ac26afe7
x=randn(T,n);

# ╔═╡ 720abc22-e9ec-48c4-a543-c83fd850b56e
function LinearAlgebra.eigvals(A₀::Arrow{T}, standardform::Bool=true,tol::Real=1e-12) where T<:Number
	# Power iteration and Wielandt deflation to compute eigenvalues of 
	# quaternionic Arrow matrix
	A=A₀
	n=size(A,1)
	# Create vector for eigenvalues
	λ=Vector{T}(undef,n)
	# First eigenpair
	λ[1],x,steps=Esolver(A,standardform,tol)
	for i=2:n-1
		# Deflated matrix
		g=x[1]\A.u[1]
		w=A.u[2:end]-x[2:end-1]*g
		α=A.α-x[end]*g
		A=Arrow(A.D[2:end],w,A.v[2:end],α,length(w)+1)
		# Eigenpair
		λ[i],x,steps=Esolver(A,standardform,tol)
	end
	# Last eigenvalue
	ν=A.α-x[2]*(x[1]\A.u[1])
	z=[one(T)]
	if standardform
		z=standardformx(ν)
	    ν=z\(ν*z)
	end
	λ[n]=ν
	return λ
end

# ╔═╡ 0b22e485-f5bd-4282-9061-4c3639019e7c
function LinearAlgebra.eigen(A₀::Arrow{T}, standardform::Bool=true,tol::Real=1e-12) where T<:Number
	# RQIds and Wielandt deflation to compute eigenvalues of 
	# quaternionic Arrow matrix
	# println("     Tolerance ",Float64(tol)," ",T)
	A=A₀
	n=size(A,1)
	# Create arrays for eigenvalues, first element and eigenvectors
	steps=zeros(Int,n)
	λ=Vector{T}(undef,n)
	γ=Vector{T}(undef,n)
	# First element of A.x
	χ=Vector{T}(undef,n)
	# First and last elements of current u
	ν=Vector{T}(undef,n)
	ψ=Vector{T}(undef,n)
	# Eigenvector matrix
	U=zeros(T,n,n)

	# First eigenvalue
	# println("Eigenvalue 1")
	λ[1],u,steps[1]=Esolver(A,false,tol)
	γ[1]=u[1]*λ[1]/u[1]
	# U[:,1]=u
	ν[1]=u[1]
	χ[1]=A.u[1]
	ψ[1]=u[n]
	for i=2:n-1
		# Deflated matrix
		g=u[1]\A.u[1]
		w=A.u[2:end]-u[2:end-1]*g
		α=A.α-u[end]*g
		A=Arrow(A.D[2:end],w,A.v[2:end],α,length(w)+1)
		# Eigenpair
		# println("Eigenvalue ",i)
		λ[i],u,steps[i]=Esolver(A,false,tol)
		γ[i]=u[1]*λ[i]/u[1]
		ν[i]=u[1]
		χ[i]=A.u[1]
		ψ[i]=u[end]
		# println(u[1],A.ρ*A.v'*u)
	end
	# Last eigenvalue
	μ=A.α-u[2]*(u[1]\A.u[1])
	z=one(T)

	λ[n]=μ
	ψ[n]=z

	# Compute the eigenvectors, bottom-up, the formulas are derived 
	# using (15) and known first and last elements of eigenvectors

	for i=n-1:-1:1
		for j=i+1:n
			ζ=sylvester(γ[i],-λ[j],χ[i]*ψ[j])
			ν[j]=ζ
			ψ[j]=ψ[j]+ψ[i]*(ν[i]\ ζ)
		end
	end
	U=eigvecs(A₀,λ,ψ)
	
	# Corrections - few steps of double-shift RQI n the original matrix using 
	# computed eigenvalues and eigenvectors
	for i=1:n
		# println(" Correction ",i)
		λ[i],U[:,i]=RQIds(A₀,λ[i],U[:,i],false, tol)
	end
		
	if standardform
		for i=1:n
			μ=λ[i]
			z=standardformx(μ)
	    	μ=z\(μ*z)
			λ[i]=μ
			U[:,i].*=z
		end
	end
	return Eigen(λ,U), sum(steps)/(n-1)
end

# ╔═╡ de95310d-ee54-4357-a5d3-ddb69e331f9e
begin
	# Arrow
	Random.seed!(5497) # 5497  5237
	Times=Vector(undef,numberofexperiments)
	TimesQR=Vector(undef,numberofexperiments)
	Errorbounds=Vector(undef,numberofexperiments)
	Errors=Vector(undef,numberofexperiments)
	ErrorsQR=Vector(undef,numberofexperiments)
	Residuals=Vector(undef,numberofexperiments)
	ResidualsB=Vector(undef,numberofexperiments)
	ResidualsQR=Vector(undef,numberofexperiments)
	Meansteps=Vector(undef,numberofexperiments)
	a=randperm(10*n)[1:n-1]
	f1(A) = [norm(A[:,i]) for i=1:size(A,2)]
	for l=1:numberofexperiments
		println("     Experiment ",l)
		# Generate the Arrow matrix
		# D₀=randn(T,n-1).* a
		D₀=[Quaternion(rand().*a[i],randn(),randn(),randn()) for i=1:n-1]
		u₀=randn(T,n-1)
		v₀=randn(T,n-1)
		# v₀=ones(T,n-1)
		α₀=randn(T)
		if matrixtype=="Hermitian"
			D₀=T.(real(D₀))
			v₀=u₀
			α₀=T(real(α₀))
		end
		global A=Arrow(D₀,u₀,v₀,α₀,n)
		# Compute the eigenvalue decomposition using fast Arrowhead eigensolver
		Times[l]=@elapsed E,mean=eigen(A,true,tol)
		Meansteps[l]=mean
		# Residuals[l]=opnorm(Matrix(A)*E.vectors-E.vectors*Diagonal(E.values),1)
		Residuals[l]=maximum(f1(Matrix(A)*E.vectors-E.vectors*Diagonal(E.values))./abs.(E.values))
		Errorbounds[l]=Residuals[l]*cond(E.vectors)
		# Eigenvalue decomposition using BigFloat()
		if n<25 
			S=T==QuaternionF64 ? Quaternion{BigFloat} : Complex{BigFloat}
			Ab=Arrow(map.(S,(A.D,A.u,A.v,A.α))...,A.i)
			Eb,meanb=eigen(Ab,true,BigFloat(1e-18))
			ResidualsB[l]=maximum(f1(Matrix(Ab)*Eb.vectors-Eb.vectors*Diagonal(Eb.values))./abs.(Eb.values))
			Es=sortperm(real(E.values))
			Ebs=sortperm(real(Eb.values))
			# Errors[l]=norm(E.values[Es]-Eb.values[Ebs],Inf)
			Errors[l]=norm(abs.(E.values[Es]-Eb.values[Ebs])./abs.(Eb.values[Ebs]),Inf)
		end
		# Eigenvalue decomposition using general eigensolver
		TimesQR[l]=@elapsed Eg=eigen(Matrix(A))
		# Residuals[l]=opnorm(Matrix(A)*E.vectors-E.vectors*Diagonal(E.values),1)
		ResidualsQR[l]=maximum(f1(Matrix(A)*Eg.vectors-Eg.vectors*Diagonal(Eg.values))./abs.(Eg.values))
		if n<25
			Egs=sortperm(real(Eg.values))
			ErrorsQR[l]=norm(abs.(Eg.values[Egs]-Eb.values[Ebs])./abs.(Eb.values[Ebs]),Inf)
		end
	end
end

# ╔═╡ 680c6d7b-e033-4583-935b-aa6d9c548b65
@time inv(A)

# ╔═╡ 9a7559d6-3d2e-4030-b01b-be92a5d5d038
@time \(A,x)

# ╔═╡ a81c0ff6-ed54-49e9-9a06-4a6b73af51a3
begin
	μ=randn(T)
	α=μ+conj(μ)
	β=μ*conj(μ)
	@time \(A,α,β,x)
end

# ╔═╡ 24f0592e-cd72-4713-b08b-553565719f86
@time eigen(A)

# ╔═╡ c283ccd8-e866-4f71-9cc0-378ca5f21eb5
@time RQIds(A)

# ╔═╡ 41331780-289f-44e8-a302-73d0f78dfe84
scatter(Meansteps)

# ╔═╡ 99ebd4df-7359-44d4-a8e2-425c0ad13f61
scatter(Times,yaxis=:log10)

# ╔═╡ 43e75924-628a-4d26-bdb6-71ebaa06962c
begin
	scatter(Errorbounds, label="Error bounds",marker=:square,mc=:green,ms=3)
	
	scatter!(Residuals,yaxis=:log10,title="Matrix=$(matrixtype), n=$(n), method=$(Esolver)",titlefontsize=12,xlabel="Experiment number",label="Residuals",mc=:red,ms=4, xticks=0:1:10)
	
	if n<30
		scatter!(Errors, label="Errors",marker=:diamond,mc=:red,ms=4)
	end
	
	scatter!(ResidualsQR,yaxis=:log10,title="Matrix=$(matrixtype), n=$(n), method=$(Esolver)",titlefontsize=12,xlabel="Experiment number",label="ResidualsQR",mc=:blue,ms=4, xticks=0:1:10)
	# ,yticks=[1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3]
	if n<30
		scatter!(ErrorsQR, label="ErrorsQR",marker=:diamond,mc=:blue,ms=4)
	end
	scatter!(legend=:topleft)
end

# ╔═╡ 36111a29-dbf8-4268-9059-1ee290fecf27
# Create filename for saving
filename=matrixtype*"_"*"$(n)"*"_"*"$(Esolver)"*"_$(-Int(log10(tol)))_.jld2"

# ╔═╡ ef341378-db5b-4745-9e74-dbd261f04972
# Save the results
@save filename Meansteps Times TimesQR Residuals ResidualsQR Errorbounds Errors ErrorsQR

# ╔═╡ 4b442542-80d8-4e12-ac92-bf65c60638c9
E,s=eigen(A)

# ╔═╡ b04cacc7-26d8-4ffd-bb7c-e3a2f93415a7
scatter([Complex(E.values[i].s,E.values[i].v1) for i=1:n])

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GenericLinearAlgebra = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Quaternions = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
GenericLinearAlgebra = "~0.3.11"
JLD2 = "~0.4.37"
Plots = "~1.39.0"
PlutoUI = "~0.7.51"
Quaternions = "~0.7.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "fced7a42fd0923df653f2b997f7d29704f35e065"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "02be7066f936af6b04669f7c370a31af9036c440"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "ebec83429b5dea3857e071d927156207ebd6d617"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.37"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "7c29f0e8c575428bd84dc3c72ece5178caa67336"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.2+2"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "49cbf7c74fafaed4c529d47d48c8f7da6a19eb75"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.1"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═ba49a550-fe54-11ed-1d8b-9b5192874bc6
# ╠═45cb51a4-9eea-4fa8-8748-8f6ba24e9c48
# ╟─41db52a5-812e-49bb-9d1f-fb7851740a88
# ╟─2552eeb1-c631-4e58-94f1-3894b9545ab9
# ╟─68b2efe8-99b0-40cf-ba11-6eb8cfa0a4d9
# ╟─8503c651-2bd3-43f2-9d4c-959962606ff5
# ╟─47f842fa-063d-4b30-a734-3f7d825b1314
# ╟─b299fcf7-7ced-45d1-a55c-74482ecb0c60
# ╟─82c8a221-3125-42a1-a033-02408688b6ae
# ╟─09fe2fc2-5ea7-428d-b0b4-f11221fbd6d3
# ╟─02afde06-04e6-44ae-b8c4-3e8e00d5b7cc
# ╟─9d38f85e-058a-426c-ae71-2f664e8357a2
# ╟─f6769f8f-19ad-47c5-a1ec-2e3c780f6cd7
# ╟─24724b41-f108-4dc7-b6dd-193a3e39bc37
# ╟─a341e994-50a7-4848-8ec8-8794a89a9063
# ╟─d3046577-b251-45b0-a743-a9970937811d
# ╟─22c35821-40f4-4c64-90b3-2ea2ce4e651c
# ╟─fa8ead94-9787-462b-9f41-47fcb41a1a17
# ╟─ff113c87-a72d-4556-98f9-e1e42782a1e6
# ╟─88e05838-d19d-45b8-b7ad-ca1fb6d47f7b
# ╟─a859ec97-d40a-4d35-908d-ccdc16c5fd57
# ╟─e747a6e4-70df-4aff-993a-e9a9ad51fa03
# ╟─cf8174d9-035d-4463-ba8b-88b1b6b44317
# ╠═3f8094c4-93ce-4b01-9b50-c7d66031a610
# ╠═9370d0a1-2ae4-4f8b-8da9-45339eeb21b4
# ╟─ddd9a82c-5d26-42be-80e1-087ba826ee10
# ╠═4e9666dc-26f7-429e-afc5-c72f14a34e9a
# ╟─dc11a4fe-c905-4cb5-a90d-8256cb469a39
# ╠═484df839-8669-4627-a3c3-919300c1c882
# ╠═23817579-826c-47fb-aeee-d67712b59ada
# ╟─bdeab9ab-4838-4332-9b74-7bba33ccb317
# ╠═53a198ac-e7bf-4dc2-9ae3-67f94f15a694
# ╟─322273eb-ca50-42b1-866a-3977700e9b63
# ╠═2c23d735-c208-4a4c-b358-cf1a12a38ebe
# ╟─124aea10-888e-438c-b0e0-82cdc1cf6dcb
# ╠═77352b2b-3642-41fe-bc8a-7046b054608b
# ╠═680c6d7b-e033-4583-935b-aa6d9c548b65
# ╠═1319217e-fa81-4188-992f-95f5ac26afe7
# ╠═9a7559d6-3d2e-4030-b01b-be92a5d5d038
# ╟─3566ab10-c23f-4f82-a4fb-2e3a134868a3
# ╠═6f71be95-f2bc-4f47-bbca-89bf8cd53cab
# ╟─1c0a18a8-8490-4790-b3b7-7f56edab3d43
# ╠═15b01358-f35b-4d43-b953-d0f046760db6
# ╟─488877d8-1dd7-43a0-97e1-ce12c2555f5d
# ╠═7570c157-1f63-47e1-9d31-c2a690e5f55b
# ╠═12178ddc-c0bc-45cd-8c45-299b3ff46029
# ╠═b909e897-bb2f-4171-9e43-caa326bae66d
# ╠═b2d0628a-35c8-4d8e-ba25-480173557229
# ╠═a81c0ff6-ed54-49e9-9a06-4a6b73af51a3
# ╠═86221eaf-9c7e-4171-b29e-66496a1a55d1
# ╠═6aa6a509-b3b3-44b4-bcad-5b0b332f3ac2
# ╟─ce762b41-6522-46d1-a332-eca6756d9687
# ╠═720abc22-e9ec-48c4-a543-c83fd850b56e
# ╟─98d415a5-1cdd-48ae-bad2-46230a7df2b9
# ╠═8d76c56a-60e8-4356-a0e5-3c41d01bc530
# ╟─7e6dcca5-e545-4822-8e49-634514fd60bb
# ╠═0b22e485-f5bd-4282-9061-4c3639019e7c
# ╟─aefbdfff-3cb2-41a4-89be-67915bfb240b
# ╠═8d04d6ae-17bf-447b-87e3-f7702fc8007c
# ╠═de95310d-ee54-4357-a5d3-ddb69e331f9e
# ╠═24f0592e-cd72-4713-b08b-553565719f86
# ╠═c283ccd8-e866-4f71-9cc0-378ca5f21eb5
# ╠═41331780-289f-44e8-a302-73d0f78dfe84
# ╠═99ebd4df-7359-44d4-a8e2-425c0ad13f61
# ╠═43e75924-628a-4d26-bdb6-71ebaa06962c
# ╠═36111a29-dbf8-4268-9059-1ee290fecf27
# ╠═ef341378-db5b-4745-9e74-dbd261f04972
# ╠═4b442542-80d8-4e12-ac92-bf65c60638c9
# ╠═b04cacc7-26d8-4ffd-bb7c-e3a2f93415a7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
