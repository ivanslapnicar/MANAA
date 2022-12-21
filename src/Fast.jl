### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 4f3a2494-e194-4bd5-9cea-5168892e7a02
# The package PlutoUI provides a Table of Contents and sliders. 
using PlutoUI

# ╔═╡ 2b4305ff-5492-4f12-a1c6-65b6d3e1be10
using Quaternions

# ╔═╡ 09729808-6c6f-4f84-a70c-ceb6b3102446
using Random

# ╔═╡ 526509a4-a5ce-4abf-b3af-d86c9029e9b3
using LinearAlgebra

# ╔═╡ 82a7b7e4-058f-42ca-8963-c0c11f3692b0
TableOfContents(title="📚 Contents", aside=true)

# ╔═╡ e7822fb1-f92c-4ac7-aba4-933070247d06
# Quaternion to 2x2 complex
function q2c(q::T) where T<:QuaternionF64
	return [complex(q.s, q.v1) complex(q.v2,q.v3);
        	complex(-q.v2,q.v3) complex(q.s,-q.v1)]
end

# ╔═╡ 0a04a062-736e-405a-8c9f-f5277a0600e9
# We need Arrow and DPR1 types
struct Arrow{T}
    D::AbstractVector{T}
    u::AbstractVector{T}
    v::AbstractVector{T}
    α::T
    i::Int
end

# ╔═╡ e4d0d2b1-0624-4d35-b78c-16a8094b1bd0
struct DPR1{T}
    Δ::Vector{T}
    x::Vector{T}
    y::Vector{T}
	ρ::T
end

# ╔═╡ e65361df-f544-4109-a45b-1a7f4ecf8c81
begin
	# rand(T,n,n) for Quaternions
	import Random.rand, Random.SamplerType
	rand(r::AbstractRNG, ::SamplerType{Quaternion{T}}) where {T<:Real} =
	    quat(rand(r, T), rand(r, T),rand(r,T),rand(r,T))
end

# ╔═╡ 70f24fa1-f225-40c9-a613-9f97ac80be9c
 Errors=Vector{Float64}(undef,8)

# ╔═╡ c358a800-e2f8-43fd-8cbc-addd1920f1ad
# This cell needs to be refreshed manually after chamgong the parameters
Errors

# ╔═╡ 91657688-ca70-47cc-a106-7d79caf07e95
begin
	# Fast multiplication
	import Base.*
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
		w[i]=adjoint(A.v)*z[ind]+A.α*zi
	    for j=A.i+1:n
	        w[j]=A.u[j-1]*zi+A.D[j-1]*z[j]
	    end
	    w
	end
	
	function *(A::DPR1,z::Vector)
	    n=size(A,1)
	    T=typeof(A.x[1])
	    w=Vector{T}(undef,n)
	    β=A.ρ*(adjoint(A.y)*z)
	    for i=1:n
	        w[i]=A.Δ[i]*z[i]+A.x[i]*β
	    end
	    w
	end

	# Multiplication (block vector) x (block)
	*(A::Vector{Matrix{T}},B::Matrix{T}) where T=[A[i]*B for i=1:length(A)]

	# Utility function to unblock the block matrix or vector
	unblock(A) = mapreduce(identity, hcat, [mapreduce(identity, vcat, A[:,i]) 
        for i = 1:size(A,2)])
end

# ╔═╡ b7e5ab61-dc73-4e30-b498-c4f5c8388704
begin
	# Arrow
	import Base.size
	size(A::Arrow, dim::Integer) = length(A.D)+1
	size(A::Arrow)= size(A,1), size(A,1)
	
	# Index into an Arrow
	import Base.getindex
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
	    end
	end
	
	# Dense version of Arrow
	import LinearAlgebra: Matrix
	Matrix(A::Arrow) =[A[i,j] for i=1:size(A,1), j=1:size(A,2)]

	# DPR1
	size(A::DPR1, dim::Integer) = length(A.Δ)
	size(A::DPR1)= size(A,1), size(A,1)
	
	# Index into DPR1
	function getindex(A::DPR1,i::Integer,j::Integer)
	    Aij=A.x[i]*A.ρ*adjoint(A.y[j])
	    return i==j ? A.Δ[i].+Aij : Aij
	end
	
	# Dense version of DPR1
	Matrix(A::DPR1)=[A[i,j] for i=1:size(A,1), j=1:size(A,2)]
end

# ╔═╡ 349ba8df-0f34-4c81-a152-5d4c771faecb
# ╠═╡ show_logs = false
md"""
# Fast multiplication, eigenvectors, determinants, and inverses of arrowhead and diagonal-plus-rank-one matrices

#### by Nevena Jakovčević Stor and Ivan Slapničar

#### from University of Split, FESB

This work has been fully supported by Croatian Science Foundation under the project IP-2020-02-2240 - [http://manaa.fesb.unist.hr](http://manaa.fesb.unist.hr).

$(PlutoUI.LocalResource("./HRZZ-eng-170x80-1.jpg"))

This notebook accompanies the paper 

> Nevena Jakovčević Stor and Ivan Slapničar, *Fast multiplication, determinants, and inverses of arrowhead and diagonal-plus-rank-one matrices over associative fields*, submitted

__All proofs are found in the paper.__

"""

# ╔═╡ 819e701a-ec3b-4c55-9b84-c6e3e8367833
md"""
# Definitions

## Quaternions

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

Let $f(x)$ be a complex analytic function. The value $f(q)$, where $q\in\mathbb{H}$, is computed by evaluating the extension of $f$ to the quaternions at $q$, see ([Sudbery,1979](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/quaternionic-analysis/308CF454034EC347D4D17D1F829F8471)). 

Quaternions are homomorphic to $\mathbb{C}^{2\times 2}$:

$$
q\to \begin{bmatrix}a+b\,\mathbf{i} & c+d\, \mathbf{i}\\-c+d\, \mathbf{i} & a-b\, \mathbf{i}\end{bmatrix}\equiv C(q),$$

with eigenvalues $q_s$ and $\bar q_s$.

Basic operations with quaternions and computation of the functions of quaternions are implemented in the package [Quaternions.jl](https://github.com/JuliaGeometry/Quaternions.jl).
"""

# ╔═╡ 48fb6f47-cae3-4739-8653-959cc9b6bd3c
md"""
## Arrowhead and DPR1 matrices

All matrices are in $\mathbb{F}^{n\times n}$ where $\mathbb{F}\in\{\mathbb{R},\mathbb{C},\mathbb{H}\}$.
$\mathbb{H}$ is a non-commutative field of quaternions.

Let $\star$ denote transpose for the real matrix and conjugate transpose (adjoint) for complex or quaternionic matrix.
  
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
"""

# ╔═╡ 348bc3dc-c4d6-40bb-a65c-5b86b4d2b1ab
md"""
# Main goal

> To use __polymorphysm__ or __multiple dispatch__ (Julia) and have the computations performed by the same code in both, commutative and noncommutative algebras.
"""

# ╔═╡ 1296b130-4c24-4602-ab9f-9d1e8080f790
md"""
For convenience, we keep errors in a single vector.

Errors[1] = Multiplication by Arrow

Errors[2] = Multiplication by DPR1

Errors[3] = Determinant of Arrow

Errors[4] = Determinant of DPR1

Errors[5] = `norm(inv(A)*A-I)` for Arrow

Errors[6] = `norm(A*inv(A)-I)` for Arrow

Errors[7] = `norm(inv(A)*A-I)` for DPR1

Errors[8] = `norm(A*inv(A)-I)` for DPR1
"""

# ╔═╡ cbb0a98a-3840-4000-99b6-78eec9d3c07f
md"""
# Types 

Number type T = $(@bind Tp Select(["Float64", "ComplexF64", "QuaternionF64"], default="Float64"))

Matrix size n = $(@bind n Slider(4:20,default=4, show_value=true))

Block size kb = $(@bind kb Slider(1:8,default=1, show_value=true))

 Index of the zero diagonal element k₀ = $(@bind k₀ Slider(0:20,default=0, show_value=true))

We have the following:

* If $kb=1$, the program generates standard (non-block) matrices. Otherwise, the program generates block-matrices with blocks of size $kb\times kb$.
* If $kb=1$ and  $k_0=0$, all diagonal elements of $D$ and $\Delta$ are non-zero.
* If $kb>1$ and  $k_0=0$, all diagonal blocks of $D$ and $\Delta$ are non-singular.
* If $kb=1$ and  $k_0>0$, then $D_{k_0,k_0}=0$ and $\Delta_{k_0,k_0}=0$. 
* If $kb>1$ and  $k_0>0$, then the blocks $D_{k_0,k_0}$ and $\Delta_{k_0,k_0}$ are zero.

"""

# ╔═╡ aeadc797-7f7f-479b-9865-0e177ab66d61
T=eval(Meta.parse(Tp))

# ╔═╡ 18eafaa0-75ea-469a-8a24-782113474386
begin
	# Random Arrow
	Random.seed!(1235)
	if kb==1
		# Standard matrix
		D=randn(T,n-1)
		u=randn(T,n-1)
		v=randn(T,n-1)
		α=randn(T)
	else
		# Block-matrix
		D=[randn(T,kb,kb) for i=1:n-1]
		u=[randn(T,kb,kb) for i=1:n-1]
		v=[randn(T,kb,kb) for i=1:n-1]
		α=randn(T,kb,kb)
	end
	# Set the tip of the arrow to the bottom right corner
	i=n
	# Set one element of the shaft to zero
	if k₀>0 & k₀<n 
		D[k₀]=zero(D[k₀])
		# D[k₀][1:3,1:3].=zero(T)
	end
	A=Arrow(D,u,v,α,i)
end

# ╔═╡ 0afceb51-1482-4deb-9544-18528af602b9
md"""
# Matrix-vector multiplication

Products $w=Az$ can be computed in $O(n)$ operations. 

Let $A=\operatorname{Arrow}(D,u,v,\alpha)$ be an arrowhead matrix with the tip at position $A_{ii}=\alpha$, and let $z$ be a vector. Then $w=Az$, where

$$
\begin{aligned}
w_j&=d_jz_j+u_jz_i, \quad i=1,2,\cdots,i-1\\
% y_i&=\sum_{j=1}^{i-1}\bar v_j x_j+\alpha x_i +\sum_{j=i}^{n-1}\bar v_j x_{j+1}\\
w_i&=v_{1:i-1}^\star z_{1:i-1} +\alpha z_i + v_{i:n-1}^\star z_{i+1:n} \\
w_j&=u_{j-1}z_i+d_{j-1}z_j,\quad j=i+1,i+2,\cdots,n.
\end{aligned}$$
Further, let $A=\operatorname{DPR1}(\Delta,x,y,\rho)$ be a DPR1 matrix and let $\beta=\rho(y^\star x) \equiv \rho (y\cdot x)$. Then $w=Az$, where

$$
w_i=\delta_i z_i+x_i\beta,\quad i=1,2,\cdots,n.$$

"""

# ╔═╡ 72d30004-b57d-4e5b-a8f3-2d49e2deb781
unblock(Matrix(A))

# ╔═╡ f543d1fc-abbe-4f97-9829-aa11dd007b61
begin
	# DPR1
	Random.seed!(5421)
	if kb==1
		# Standard matrix
		Δ=randn(T,n)*10
		x=randn(T,n)
		y=randn(T,n)
		ρ=randn(T)
	else
		# Block matrix
		Δ=[randn(T,kb,kb) for i=1:n]
		x=[randn(T,kb,kb) for i=1:n]
		y=[randn(T,kb,kb) for i=1:n]
		ρ=randn(T,kb,kb)
	end
	# Set one element of Δ to zero
	if k₀>0 & k₀<=n 
		Δ[k₀]=zero(Δ[k₀])
	end
	B=DPR1(Δ,x,y,ρ)
end

# ╔═╡ 035a716e-b3d5-4450-bc03-f8663ed45e93
Matrix(B)

# ╔═╡ c163f955-ce07-43d6-9166-f53164c3d40d
	# Generate the vector z
	if kb==1
		z=rand(T,n)
	else
		z=[randn(T,kb,kb) for i=1:n]
	end

# ╔═╡ 91a112a5-de47-44f4-a2ea-d45eef1fc43e
md"
Functions `norm()`, `det()` and `inv()` from the package `LinearAlgebra` are used for testing. 
"

# ╔═╡ 76d9174a-cdc8-4837-80ac-ff8f066a7cca
# Check multiplication by Arrow
Errors[1]=norm(unblock(A*z)-unblock(Matrix(A))*unblock(z))

# ╔═╡ 56489d74-e38c-4d87-9c06-35ed26206714
# Check multiplication by DPR1
Errors[2]=norm(unblock(B*z)-unblock(Matrix(B))*unblock(z))

# ╔═╡ d3acdefb-dcf1-42e3-a4b7-066b7640b622
md"""
# Determinants

Determinants are computed using two basic facts:
* the determinant of the triangular matrix is a product of diagonal elements ordered from the first to the last, and
* the determinant of the product is the product of determinants.

Here are some facts:

* The determinants are computed in $O(n)$ operations.
* For the proofs see the paper.
* For block matrices and quaternionic matrices, due to non-commutativity, the computations must be carried out exactly in the order specified.
* The determinant of the matrix of quaternions can be defined using a determinant of its corresponding homomorphic complex matrix. For matrices of quaternions, the determinant is not well defined due to non-commutativity. However, the Study determinant, $|\det(A)|$, is well defined ([Aslaksen, 1996]((https://www.researchgate.net/profile/Helmer-Aslaksen/publication/226528876_Quaternionic_Determinants/links/0046352f74a71a77f2000000/Quaternionic-Determinants.pdf))). Therefore, after computing the respective determinant, it suffices to take the absolute value.
* For block matrices with $k\times k$ blocks, the formulas return a $k\times k$ matrix, so one more step is required -- computing the determinant of this matrix.


## Arrowhead

Let $A=\operatorname{Arrow}(D,u,v,\alpha)$ be an arrowhead matrix. If all $d_i\neq 0$, the determinant of $A$ is equal to

$$
\det(A)= (\prod_i d_i)(\alpha-v^\star D^{-1}u). $$

If $d_i=0$, then 

$$
\det(A)=\bigg(\prod_{j=1}^{i-1}d_j\bigg)\cdot v_i^\star 
\cdot \bigg(\prod_{j=i+1}^{n-1}d_j\bigg)\cdot  u_i.$$

## DPR1

Let $A=\operatorname{DPR1}(\Delta,x,y,\rho)$ be a DPR1 matrix. If all $\delta_i\neq 0$, the determinant of $A$ is equal to

$$
\det(A)=(\prod_i \delta_i)(1+y^\star \Delta^{-1}x\rho).$$

If $\delta_i=0$, then 

$$
\det(A)=(\prod_{j=1}^{i-1} \delta_j) y_i^\star (\prod_{j=i+1}^n \delta_j)x_i \rho.$$

"""

# ╔═╡ 68284dc0-f4f9-403f-9dd9-02ec245726a6
md"
__Check special cases on examples from papers in the `tex` directory.__

__Important__ Check the case when $\Delta_{ii}+x_i \rho \overline{y_j}=0$ for complex or quaternionic matrices.
"

# ╔═╡ 33b5862b-ddf0-4783-aa9b-6f280f5b162e
begin
	import LinearAlgebra.det
	function det(A::Arrow)
	    i=findfirst(iszero.(A.D))
	    if i==nothing
	        d=prod(A.D)*(A.α-adjoint(A.v)*(A.D .\A.u))
	    else
			if i<length(A.D)
	        	d=-prod(A.D[1:i-1])*adjoint(A.v[i])*prod(A.D[i+1:end])*A.u[i]
			else
				d=-prod(A.D[1:i-1])*adjoint(A.v[i])*A.u[i]
			end
		end
		# For the block matrix, compute the determinant
		d=isa(A.D[1],Array) ? det(d) : d
		# For the quaternionic matrix, take the absolute value
		return T==QuaternionF64 ? abs(d) : d
	end

	function det(A::DPR1)
	    i=findfirst(iszero.(A.Δ))
	    if i==nothing
	        d=prod(A.Δ)*(one(A.Δ[1])+adjoint(A.y)*(A.Δ .\ (A.x*A.ρ)))
	    else
			if i<n
		        d=prod(A.Δ[1:i-1])*adjoint(A.y[i])*prod(A.Δ[i+1:end])*(A.x[i]*A.ρ)
			else
				d=prod(A.Δ[1:i-1])*adjoint(A.y[i])*(A.x[i]*A.ρ)
			end
		end
		# For the block matrix, compute the determinant
		d=isa(A.Δ[1],Array) ? det(d) : d
		# For the quaternionic matrix, take the absolute value
		return T==QuaternionF64 ? abs(d) : d
	end

	# We need a simple quaternionic determinant based on homomorphism to C
	function det(A::Matrix{T}) where T<:QuaternionF64
		return sqrt(abs(det(unblock(q2c.(A)))))
	end
end

# ╔═╡ 16e951e0-950a-44ce-92b8-d7aa83791a3b
# Arrow
det(A)

# ╔═╡ 1de39091-4896-4810-a9ac-b99cbb7bd02a
Errors[3]=det(A)-det(unblock(Matrix(A)))

# ╔═╡ f19eca78-1cd4-4698-bae1-6c77111edcd1
# DPR1
det(B)

# ╔═╡ 941511ed-4cb2-4e24-82a1-1ced511880fa
Errors[4]=det(B)-det(unblock(Matrix(B)))

# ╔═╡ db93dd38-416a-4028-9261-f469a5b81743
md"""
# Inverses

Let $\dagger$ denote the inverse of a scalar and the pseudo-inverse of a matrix.

For matrices of quaternions, the computation of pseudo-inverse is implemented in the package [GenericLinearAlgebra.jl](https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl).

## Arrow 
Let $A$ be a non-singular Arrow with the tip at the position $A_{ii}=\alpha$ and let $P$ be the permutation matrix of the permutation $p=(1,2,\cdots,i-1,n,i,i+1,\cdots,n-1)$. 

If all $d_j\neq 0$, the inverse of $A$ is a DPR1 matrix

$$
A^{-1} =\Delta+x \rho y^⋆,$$

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
A^{-1}=P\begin{bmatrix} \hat D & \hat u\\ \hat v^\star & \hat \alpha \end{bmatrix}P^T,
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

# ╔═╡ 3d9476ed-1967-4d13-8d02-78586f20cea1
md"""

## DPR1

Let $A$ be a non-singular DPR1 matrix. 

If all $\delta_j\neq 0$, the inverse of $A$ is a DPR1 matrix

$$
A^{-1} =\hat\Delta+\hat x\hat \rho \hat y^*,$$

where 

$$
\hat \Delta=\Delta^{-1},\quad 
\quad \hat x=\Delta^{-1}x,\quad
\hat y=\Delta^{-*}y,\quad
\hat \rho=-\rho(I-y^* \Delta^{-1} x\rho)^{-1}.$$


If $\delta_j=0$, the inverse of $A$ is an arrowhead matrix with the tip of the arrow at position $(j,j)$. In particular, let $P$ be the permutation matrix of the permutation  $p=(1,2,\cdots,j-1,n,j,j+1,\cdots,n-1)$. Partition $\Delta$, $x$ and $y$ as

$$
\Delta=\begin{bmatrix}\Delta_1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & \Delta_2\end{bmatrix},\quad
x=\begin{bmatrix} x_1 \\ x_j \\x_2\end{bmatrix},\quad 
y=\begin{bmatrix} y_1 \\ y_j \\y_2\end{bmatrix}.$$

Then, 

$$
A^{-1}=P\begin{bmatrix} D & u \\ v^T & \alpha \end{bmatrix}P^T,$$

where

$$
\begin{align*}
D&=\begin{bmatrix} \Delta_1^{-1} & 0\\ 0 &\Delta_2^{-1}\end{bmatrix},\quad
u= \begin{bmatrix}-\Delta_1^{-1}x_1 \\ -\Delta_2^{-1}x_2\end{bmatrix} x_j^{-1},\quad
v= \begin{bmatrix}-\Delta_1^{-\star}y_1 \\ -\Delta_2^{-\star}y_2\end{bmatrix}y_j^{-1},\\
\alpha&=(y_j^{-1})^\star\left(\rho^{-1} +y_1^\star \Delta_1^{-1} x_1+y_2^\star \Delta_2^{-1}x_2\right) x_j^{-1}.
\end{align*}$$

> The formulas for inverses hold for quaternionic and block matrices!

"""

# ╔═╡ a26e610d-f9cf-4e95-97ac-cfcd84836d34
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
	        return DPR1(Δ[p],x[p],y[p],ρ)
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

	function inv(A::DPR1)
    	j=findfirst(iszero.(A.Δ))
		n=length(A.Δ)
    	if j==nothing
			Δₕ=inv.(A.Δ)
        	xₕ=Δₕ .* A.x
        	yₕ=adjoint.(Δₕ) .* A.y
        	
        	ρₕ=-A.ρ*inv(I+adjoint(A.y)*(Δₕ .* (A.x*A.ρ)))
        	return DPR1(Δₕ,xₕ,yₕ,ρₕ)
    	else
        	ind=[1:j-1;j+1:n]
        	Δ=inv.(A.Δ[ind])
        	x=A.x[ind]
        	y=A.y[ind]
        	uₕ=(-Δ .* x)*inv(A.x[j])
        	vₕ=(-adjoint.(Δ) .* y)*inv(A.y[j])
        	αₕ=adjoint(inv(A.y[j]))*(inv(A.ρ)+adjoint(y)*(Δ .* x)) *inv(A.x[j])   
    	    return Arrow(Δ,uₕ,vₕ,αₕ,j)
    	end
	end
end

# ╔═╡ 701de988-2523-4468-8f9a-d61021fbd52d
# Arrow
C=inv(A)

# ╔═╡ d43180c2-a607-4044-9cc8-f7f67e59be06
Matrix(A)

# ╔═╡ 0771f810-a749-4bcc-b971-c01dbaa44ab6
Matrix(C)

# ╔═╡ 6212c974-bc48-4be3-99ab-0e904578af04
norm(inv(unblock(Matrix(A)))*unblock(Matrix(A))-I)

# ╔═╡ 85711df7-0040-4fff-abb8-00353bb325e0
Errors[5]=norm(Matrix(C)*Matrix(A)-I); Errors[6]=norm(Matrix(A)*Matrix(C)-I)

# ╔═╡ f3d02bb0-084a-4e87-8119-eb613c348ea4
# DPR1
F=inv(B)

# ╔═╡ a8db62c6-a96e-459f-8214-a849f4e4faeb
Errors[7]=norm(Matrix(F)*Matrix(B)-I); Errors[8]=norm(Matrix(B)*Matrix(F)-I)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Quaternions = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
PlutoUI = "~0.7.30"
Quaternions = "~0.6.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "031e0ce7a8d35b9c39261dca24f6a1e714e61fb4"

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

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

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
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "92f91ba9e5941fc781fecf5494ac1da87bdac775"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5c0eb9099596090bb3215260ceca687b888a1575"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.30"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "fd78cbfa5f5be5f81a482908f8ccfad611dca9a9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.6.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
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
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═4f3a2494-e194-4bd5-9cea-5168892e7a02
# ╠═82a7b7e4-058f-42ca-8963-c0c11f3692b0
# ╟─349ba8df-0f34-4c81-a152-5d4c771faecb
# ╟─819e701a-ec3b-4c55-9b84-c6e3e8367833
# ╠═2b4305ff-5492-4f12-a1c6-65b6d3e1be10
# ╠═e7822fb1-f92c-4ac7-aba4-933070247d06
# ╟─48fb6f47-cae3-4739-8653-959cc9b6bd3c
# ╠═0a04a062-736e-405a-8c9f-f5277a0600e9
# ╠═e4d0d2b1-0624-4d35-b78c-16a8094b1bd0
# ╠═b7e5ab61-dc73-4e30-b498-c4f5c8388704
# ╟─348bc3dc-c4d6-40bb-a65c-5b86b4d2b1ab
# ╠═09729808-6c6f-4f84-a70c-ceb6b3102446
# ╠═e65361df-f544-4109-a45b-1a7f4ecf8c81
# ╟─1296b130-4c24-4602-ab9f-9d1e8080f790
# ╠═70f24fa1-f225-40c9-a613-9f97ac80be9c
# ╟─cbb0a98a-3840-4000-99b6-78eec9d3c07f
# ╠═c358a800-e2f8-43fd-8cbc-addd1920f1ad
# ╠═aeadc797-7f7f-479b-9865-0e177ab66d61
# ╠═72d30004-b57d-4e5b-a8f3-2d49e2deb781
# ╠═18eafaa0-75ea-469a-8a24-782113474386
# ╠═f543d1fc-abbe-4f97-9829-aa11dd007b61
# ╠═035a716e-b3d5-4450-bc03-f8663ed45e93
# ╟─0afceb51-1482-4deb-9544-18528af602b9
# ╠═91657688-ca70-47cc-a106-7d79caf07e95
# ╠═c163f955-ce07-43d6-9166-f53164c3d40d
# ╟─91a112a5-de47-44f4-a2ea-d45eef1fc43e
# ╠═526509a4-a5ce-4abf-b3af-d86c9029e9b3
# ╠═76d9174a-cdc8-4837-80ac-ff8f066a7cca
# ╠═56489d74-e38c-4d87-9c06-35ed26206714
# ╟─d3acdefb-dcf1-42e3-a4b7-066b7640b622
# ╟─68284dc0-f4f9-403f-9dd9-02ec245726a6
# ╠═33b5862b-ddf0-4783-aa9b-6f280f5b162e
# ╠═16e951e0-950a-44ce-92b8-d7aa83791a3b
# ╠═1de39091-4896-4810-a9ac-b99cbb7bd02a
# ╠═f19eca78-1cd4-4698-bae1-6c77111edcd1
# ╠═941511ed-4cb2-4e24-82a1-1ced511880fa
# ╟─db93dd38-416a-4028-9261-f469a5b81743
# ╟─3d9476ed-1967-4d13-8d02-78586f20cea1
# ╠═a26e610d-f9cf-4e95-97ac-cfcd84836d34
# ╠═701de988-2523-4468-8f9a-d61021fbd52d
# ╠═d43180c2-a607-4044-9cc8-f7f67e59be06
# ╠═0771f810-a749-4bcc-b971-c01dbaa44ab6
# ╠═6212c974-bc48-4be3-99ab-0e904578af04
# ╠═85711df7-0040-4fff-abb8-00353bb325e0
# ╠═f3d02bb0-084a-4e87-8119-eb613c348ea4
# ╠═a8db62c6-a96e-459f-8214-a849f4e4faeb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
