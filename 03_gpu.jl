### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 22cec81f-c503-43de-8297-ec0591e4e6c5
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("Plots")
		Pkg.add("BenchmarkTools")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using Plots
	using BenchmarkTools
	using DistanceTransforms
end

# ╔═╡ ed7fea36-60ca-47e0-9046-bf2565fdd41d
md"""
## Import packages
"""

# ╔═╡ afeb0802-60c1-40e0-a9e5-b13ad6af27bc
TableOfContents()

# ╔═╡ 4f2e365c-6e11-490e-933f-c9036ae4e018
md"""
## GPU enabled distance transforms
One of the advantages of a Julia-based distance transform library is how accessible something like GPU programming is for complex algorithms. The `SquaredEuclidean` distance transform is highly parallelizable and is set up to take advantage of the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) library and GPU hardware.

Unfortunately, [glassnotebook.io](https://glassnotebook.io) does not yet support GPU compatible static exports, so this documentation, as it stands currently, only shows how to set up a GPU-based distance transform, without any execution.

The code is copy-pastable though for anyone interested in testing it out on their own GPU-enabled computer.
"""

# ╔═╡ f94cf7fa-91de-4fbe-ae52-2063bd01f2ba
md"""
```julia
array = CUDA.CuArray(
	boolean_indicator(
		[
			0 1 1 1 0
			1 1 1 1 1
			1 0 0 0 1
			1 0 0 0 1
			1 0 0 0 1
			1 1 1 1 1
			0 1 1 1 0
		]
	)
)

dt = CuArray{Float32}(undef, size(x))
v = CUDA.ones(Int64, size(x))
z = CUDA.zeros(Float32, size(x) .+ 1)
tfm = DistanceTransforms.SquaredEuclidean(x, dt, v, z)

transform!(x, tfm)
```

This would output a CuArray, like so:
```julia
CUDA.CuArray(
	[
		1.0 0.0 0.0 0.0 1.0
		0.0 0.0 0.0 0.0 0.0
		0.0 1.0 1.0 1.0 0.0
		0.0 4.0 4.0 4.0 0.0
		0.0 1.0 1.0 1.0 0.0
		0.0 0.0 0.0 0.0 0.0
		1.0 0.0 0.0 0.0 1.0
	]
)
```
"""

# ╔═╡ 83901aae-6c82-4d4e-b85d-cfb8fbefa23c
md"""
## GPU enabled distance transform use cases

Distance transforms are ubiquitous in the fields of computer vision and image processing. From object recognition and path planning to deep learning and image segmentation, distance transforms are increasingly useful. DistanceTransforms.jl was created to give developers a simple, consistent API for implementing various distance transforms and to give end-users a seamless way to utilize distance transforms for various tasks, especially GPU-related tasks.

One such example can be seen in the next tutorial on using distance transforms within various [loss functions](link...)
"""

# ╔═╡ Cell order:
# ╟─ed7fea36-60ca-47e0-9046-bf2565fdd41d
# ╠═22cec81f-c503-43de-8297-ec0591e4e6c5
# ╠═afeb0802-60c1-40e0-a9e5-b13ad6af27bc
# ╟─4f2e365c-6e11-490e-933f-c9036ae4e018
# ╟─f94cf7fa-91de-4fbe-ae52-2063bd01f2ba
# ╟─83901aae-6c82-4d4e-b85d-cfb8fbefa23c
