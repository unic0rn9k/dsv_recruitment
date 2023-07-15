### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ d3f709bc-bbe7-46a9-b49c-f51e70241ad1
begin
	### A Pluto.jl notebook ###
	# v0.19.26
	
	import Pkg
	Pkg.add([
	    "Images",
	    "Plots",
	    "StaticArrays",
	    "Distributions",
	    "Statistics",
	    "JSON",
	    "PlotlyJS"
	])
	
	using Markdown
	using InteractiveUtils
	using Images
	using Plots
	plotly()
	using Distributions
	using StaticArrays
	using Random
	using Serialization
	using NearestNeighbors
	using JSON
	using PlotlyJS
	import .Iterators.flatten
end

# ╔═╡ 424e68b9-4fb3-4a1f-b0ed-3dd759038c76
md"""
# Notebook for DSV recruitment test

First I just import some basic libraries...
"""

# ╔═╡ 13bc5f6e-a9ab-4692-93fa-fbc8d18e6e22
md"""
Chronologically, I started by writing some functions for loading the training and test data. Just is a good starting point, because it got me thinking about how the data was structured, which also helped me get ideas for how to perseed when doing data exploration.
"""

# ╔═╡ d17f94c7-ead8-4eeb-ac7f-2cde8f057dcc
begin
	training_data_path = "./data science part dataset/dataset/training_data/"
	test_data_path = "./data science part dataset/dataset/testing_data/"
	labels = ["question", "answer", "header", "other"]
	IDs = (x->x[1:end-4]).(readdir("$training_data_path/images"))
	test_IDs = (x->x[1:end-4]).(readdir("$test_data_path/images"))
end

# ╔═╡ 2c38bdd6-771f-476e-af1f-1126a82b73dd
function trainig_data(n)
	image = Images.load("$training_data_path/images/$(IDs[n]).png")
	json = JSON.parsefile("$training_data_path/annotations/$(IDs[n]).json")
	return (image, json)
end

# ╔═╡ b934f159-6f29-4add-a5e4-23cbdb4bf5ed
function test_data(n)
	image = nothing#Images.load("$test_data_path/images/$(test_IDs[n]).png")
	json = JSON.parsefile("$test_data_path/annotations/$(test_IDs[n]).json")
	return (image, json)
end

# ╔═╡ c38cc8b0-26b0-41a4-a957-38f447f6c471
function box(image, annotation)#::(String, String, Matrix, Array)
	n = annotation["box"]
	local i
	try
		i = image[n[2]:n[4],n[1]:n[3]]
	catch _
		i = []
	end
	return (annotation["label"], annotation["text"], i, n, annotation["linking"])
end

# ╔═╡ 6613d8ec-ab7e-4ac9-b92a-e0c1b674e84c
function boxes(data)#::Vector{Tuple{String, String, Matrix{Gray{N0f8}}, Vector{Any}}}
	image = data[1]
	json = data[2]

	[box(image, a) for a in json["form"]]
end

# ╔═╡ a496ba66-d72e-45bf-a431-cb06024bb939
md"""
Julia makes it easy to look at the data. Even tho I did have a look at the raw JSON at first. I never actually use the cropped image boxes, but I think they add some visual clarity, and they where nice when I was debugging, so I decided to keep them.
"""

# ╔═╡ e66178d7-5822-42d2-8810-b26bdc8ad83a
boxes(trainig_data(2))

# ╔═╡ ddb3bd70-3f6d-4344-8010-3e253f8e425b
md"""
After looking a bit at the data, I notice that a lot of questions end on the charecter ':', and there seam to be a lot of questions in general, so I collect some data, and plot some graphs to confirm this.

I also made some naive handcrafted classifiers, to set a baseline for how I could expect a reasonable classifier to performe.
Since the questions make up around 44% of the data, just betting on that is pretty safe. And a policy predicting 'question' for all strings ending on ':', and 'answer' (the next largest majority of the labels) for anything that does't, will result in somthing close to a 50% accuracy.
"""

# ╔═╡ ffe2bbf4-b586-40c6-9019-bd10046446fc
begin
	# How often specific strings occur for a label type	
	frequency = Dict("question"=>0.0, "answer"=>0, "header"=>0, "other"=>0)
	avg_size = Dict("question"=>[0,0,0], "answer"=>[0,0,0], "header"=>[0,0,0], "other"=>[0,0,0])
	avg_pos = Dict("question"=>[0,0,0], "answer"=>[0,0,0], "header"=>[0,0,0], "other"=>[0,0,0])
	question_char = Dict(""=>0)
	
	for td in 1:length(IDs)
		for box in boxes(trainig_data(td))
			frequency[box[1]]+=1
			if box[1] == "question" && box[2] != ""
				idx = string(box[2][end])
				old = get!(question_char, idx, 0)
				question_char[idx] = old+1
			end
	
			avg_size[box[1]] += [box[4][3]-box[4][1], box[4][4]-box[4][2],1]
			avg_pos[box[1]] += [box[4][1], box[4][2], 1]
		end
	end
	
	qkeys::Array{String} = [k for k in keys(question_char)]
	qvalues::Array{Int64} = [question_char[i] for i in qkeys]
	
	maxx::Array{Int64} = partialsortperm(qvalues, 1:20, rev=true)
	first = [(qkeys[i], qvalues[i]) for i::Int64 in maxx[1:4]]
	then = sum([v for (i,v) in enumerate(qvalues) if !(i in maxx)])
	question_char = vcat(first, [("andre", then)])
	
	fsums = sum(values(frequency))
	for key in keys(frequency)
		frequency[key] /= fsums
	end
end

# ╔═╡ ea56396a-72ff-4b63-a75c-ecc60daa637e
Plots.bar(frequency)

# ╔═╡ d42e436c-9a9d-490e-ab50-f273a10766cc
Plots.bar(question_char)

# ╔═╡ a66ae78a-5c37-4464-a93c-dcf91ed62ad0
md"""
After having tried a couple of different aproaches, I settle on using a bayes classifier, ensebled with a KNN.

The bayes classifier uses a weighted combination of a Gausian (for the continous data, such as box position and size), and a Bernoulli classifier. Originally I had used a regular multinomial bayes classifier, but the Bernoulli (It just counts the tokens as having occured in a string, or not, instead of the amount of occurences) tourned out to work better.
"""

# ╔═╡ f79d8474-e0a8-49b0-9e0b-93ec573240d4
struct BayesClass
	words::Dict{String, UInt64}
	locations::Array{Array{UInt64}}
	sizes::Array{Array{UInt64}}
end

# ╔═╡ 58d4f627-0232-40b5-8481-0953c81278b0
function tokenize(text_::String)::Array{String}
	text::String = text_
	
	for char in [':', '☐', '☑', ',', '.', '-', '$', '/', '"', '\'', '%', '@', ';', '[', ']', '+', '{', '}', '(' ,')', '<', '>']
		text = replace(text, string(char)=>" $char ")
	end
	
	tokens::Dict{String, Nothing} = Dict()
	for token in split(text," ")
		if token != ""
			get!(tokens,string(token),nothing)
		end
	end
	[t for t in keys(tokens)]
end

# ╔═╡ 716729de-bd39-44e0-9408-740ffd3522fe
tokenize("ok boomer:")

# ╔═╡ 299a5e01-fd3c-4dba-ad08-b41ca2ae0a3f
begin
	bayes_table::Dict{String, BayesClass} = Dict()
	for label in labels
		get!(bayes_table, label, BayesClass(Dict(),[],[]))
	end

	for td in 1:length(IDs)
		for box in boxes(trainig_data(td))
			push!(bayes_table[box[1]].locations, box[4][1:2])
			push!(bayes_table[box[1]].sizes, abs.(box[4][1:2]-box[4][3:4]))
			
			for word in tokenize(box[2])
				bay = get!(bayes_table[box[1]].words, word, 0)
				bayes_table[box[1]].words[word] = bay + 1
			end
		end
	end
end

# ╔═╡ 711ba79a-0621-412c-8914-422d2a8926ac
Label = SVector{4, Float64}

# ╔═╡ b10b21fd-c6bc-4e0e-a2e6-30684027e53c
function naive_bayes(text::String, box::Array{Any})::Label
	pc::Array{Float64} = [frequency[l] for l in labels]
	size = abs.(box[1:2]-box[3:4])
	p::Array{Float64} = zeros(4)
	
	for l in 1:length(p)
		sizes_x = fit(Normal, [p[1] for p in bayes_table[labels[l]].sizes])
		sizes_y = fit(Normal, [p[2] for p in bayes_table[labels[l]].sizes])
		poses_x = fit(Normal, [p[1] for p in bayes_table[labels[l]].locations])
		poses_y = fit(Normal, [p[2] for p in bayes_table[labels[l]].locations])

		p[l] += (log(pdf(poses_x, box[1])) +
			log(pdf(poses_y, box[2]))) * 0.7 +
			(log(pdf(sizes_x, size[1])) +
			log(pdf(sizes_y, size[2]))) * 0.3
	end
		
	
	for word in tokenize(text)
		pw =
			log(sum([get(bayes_table[labels[l]].words, word, 0)+1 for l in 1:length(p)])) -
			log(sum([ sum(values(bayes_table[labels[l]].words)) for l in 1:length(p)]))
		
		for l in 1:length(p)
			p[l] += log(get(bayes_table[labels[l]].words, word, 0)+1) - log(sum(values(bayes_table[labels[l]].words))+1) + log(pc[l]) - pw
		end
	end
	
	return p
end

# ╔═╡ ed2d9c77-ed76-4391-b0e5-681b4c2faa54
begin
	local bacc = 0
	local biter = 0

	for td in 1:length(test_IDs)
		for box in boxes(test_data(td))
			biter+=1
			if labels[argmax(naive_bayes(box[2], box[4]))] == box[1]
				bacc+=1
			end
		end
	end

	local accuracy = bacc*100/biter
	println("The Bayes classifier has a $accuracy% on test set")
end

# ╔═╡ 7901326c-655f-47af-8e78-4e8e1034bc95
#softmax(x::Vector{Float64}) = exp.(x) / sum(exp.(x))
function softmax(x::StaticVector{D, Float64})::SVector{D, Float64} where D
    exps = exp.(x .- maximum(x))
    exps / sum(exps)
end

# ╔═╡ 3e8530d7-7e0f-40fb-a165-b5806d8a3844
struct Sibling
	direction::Vector{Float64}
	index::UInt
end

# ╔═╡ 8daad32e-35c1-4ffd-92b5-5bba9748736b
struct Box{T}
	data::Tuple{String, String, T, Vector{Any}, Vector{Any}}
	siblings::Array{Sibling}
	bayes_label::Label
end

# ╔═╡ 01cc967f-6a18-4cda-8da3-f48ea4265e3c
Embedding = SVector{20, Float64}

# ╔═╡ 981829e2-fc92-4d97-8cac-c2272f91ee40
normalize(v) = if all(iszero, v); v; else v / norm(v) end

# ╔═╡ 98102564-0b2b-454b-8b2d-1a3c2cde362b
function target(label::String)::Label
	[if label == l
		1.0
	else
		0.0
	end for l in labels]
end

# ╔═╡ a13cd725-56e6-4f6d-a923-ee858856e3e4
function shortest_distance(rect1::Vector{A}, rect2::Vector{B}) where {A,B}
    x = min(abs(rect1[1]-rect2[3]), abs(rect1[3]-rect2[1]), abs(rect1[3]-rect2[3]), abs(rect1[1]-rect2[1]))
    y = min(abs(rect1[2]-rect2[4]), abs(rect1[4]-rect2[2]), abs(rect1[4]-rect2[4]), abs(rect1[2]-rect2[2]))
    
    return [x, y] .* sign.(rect2[1:2] .- rect1[1:2])
end

# ╔═╡ 8a1ee585-5a1e-42b1-b1f2-fbd7a959b564
md"""
The `train_page` function builds a graph of a PDF page, from the training data, taking the type returned by the `boxes` function as an input.
"""

# ╔═╡ 0c3b9ab8-5aa7-4fce-acb2-3761f1dc8650
function train_page(data::Vector{Tuple{String, String, T, Vector{Any}, Vector{Any}}}; test::Bool=true)::Array{Box} where T
	bayes_labels = if test
		bayes_labels = [softmax(naive_bayes(dp[2], dp[4])) for dp in data]
	else
		[target(dp[1]) for dp in data]
	end

	[begin
		sibs::Array{Sibling} = [
			Sibling(if dp2[4] != dp[4]
				shortest_distance(dp2[4], dp[4])
			else
				[Inf, Inf]
			end, i)
		for (i, dp2) in enumerate(data)]

		sort!(sibs, by= x-> sqrt(sum(x.direction.^2)))

		Box(dp, sibs, bayes_labels[ndp])
	end for (ndp, dp) in enumerate(data)]
end

# ╔═╡ 4148fac3-f834-432c-833e-7c5005cb52d5
md"""
The embedding of a box, consists of the bayes predicted label, and a flattened embeddings for the neighboring boxes, stored in the `sib_data` variable.

Here `sib_data` roughly represents the probability of finding a label in one of four directions, parralle to the x,y axies. This is encodet as a matrix, where the rows represent each of the 4 directins, and the columns represent each of the labels.
The labels used in the `sib_data` embedding, also comes from the bayes classifier.
"""

# ╔═╡ 79cfa429-7668-45cb-8691-ccc59238c09a
function embedding(page::Vector{Box}, box_id::Int64)::Embedding
	box = page[box_id]
	sib_data::MMatrix{4,4,Float64} = zeros(4,4)
	sibs = box.siblings

	dirs = [[1,0],[0,1], [-1,0],[0,-1]]
		
	for (j, dir) in enumerate(dirs)
		for sib in sibs
			scale = max(0,dot(normalize(sib.direction), dir)) / sqrt(sum(sib.direction.^2))
			
			n = page[sib.index].bayes_label * scale
			if !any((x->isinf(x) || isnan(x)), n)
				sib_data[j,:] += n
			end
		end
		sib_data[j,:] = softmax(ℯ.^sib_data[j,:])
	end
	
	vcat(collect(flatten(vcat(sib_data))), softmax(naive_bayes(box.data[2], box.data[4])) * 0.2)
end

# ╔═╡ 22932754-adb1-4a11-810a-59e72980d16a
md"""
# KNN

Firs I build the KNN model...
"""

# ╔═╡ 4a462f59-57d1-4b63-a281-1a82bf250e54
begin
	knn_data::Vector{Embedding} = []
	knn_labels::Vector{Label} = []
	
	for td in 1:length(IDs)
	    page = train_page(boxes(trainig_data(td)))
	    for box in 1:length(page)
	        input = vcat(embedding(page, box))
	        push!(knn_data, input)
	        push!(knn_labels, target(page[box[1]].data[1]))
	    end
	end
	
	kdtree = KDTree(knn_data, Minkowski(0.2))
end

# ╔═╡ ac644379-6ffa-45d1-a95b-0b3ae522e6a7
md"""
Then I test it on the test set...
"""

# ╔═╡ 8ac77806-9ae7-43f9-b644-edb4a2d9d38d
begin
	local acc = 0
	local iter = 0
	for td in 1:length(test_IDs)
	    page = train_page(boxes(test_data(td)))
	    for box in 1:length(page)
	        iter+=1
	        test_dp = vcat(embedding(page, box))
	
	        s = knn(kdtree, test_dp, 10, true)
	        sorted = [knn_labels[a]/b for (a,b) in zip(s[1], s[2])]
	        if argmax(sum(sorted)) == argmax(target(page[box].data[1]))
	            acc+=1
	        end
	    end
	end
	println("The KNN and Bayes ensemble has a $(acc*100/iter)% accuracy on the test set")
end

# ╔═╡ Cell order:
# ╟─424e68b9-4fb3-4a1f-b0ed-3dd759038c76
# ╠═d3f709bc-bbe7-46a9-b49c-f51e70241ad1
# ╟─13bc5f6e-a9ab-4692-93fa-fbc8d18e6e22
# ╠═d17f94c7-ead8-4eeb-ac7f-2cde8f057dcc
# ╠═2c38bdd6-771f-476e-af1f-1126a82b73dd
# ╠═b934f159-6f29-4add-a5e4-23cbdb4bf5ed
# ╠═c38cc8b0-26b0-41a4-a957-38f447f6c471
# ╠═6613d8ec-ab7e-4ac9-b92a-e0c1b674e84c
# ╟─a496ba66-d72e-45bf-a431-cb06024bb939
# ╠═e66178d7-5822-42d2-8810-b26bdc8ad83a
# ╟─ddb3bd70-3f6d-4344-8010-3e253f8e425b
# ╠═ffe2bbf4-b586-40c6-9019-bd10046446fc
# ╠═ea56396a-72ff-4b63-a75c-ecc60daa637e
# ╠═d42e436c-9a9d-490e-ab50-f273a10766cc
# ╟─a66ae78a-5c37-4464-a93c-dcf91ed62ad0
# ╠═f79d8474-e0a8-49b0-9e0b-93ec573240d4
# ╠═58d4f627-0232-40b5-8481-0953c81278b0
# ╠═716729de-bd39-44e0-9408-740ffd3522fe
# ╠═299a5e01-fd3c-4dba-ad08-b41ca2ae0a3f
# ╠═711ba79a-0621-412c-8914-422d2a8926ac
# ╠═b10b21fd-c6bc-4e0e-a2e6-30684027e53c
# ╠═ed2d9c77-ed76-4391-b0e5-681b4c2faa54
# ╠═7901326c-655f-47af-8e78-4e8e1034bc95
# ╠═3e8530d7-7e0f-40fb-a165-b5806d8a3844
# ╠═8daad32e-35c1-4ffd-92b5-5bba9748736b
# ╠═01cc967f-6a18-4cda-8da3-f48ea4265e3c
# ╠═981829e2-fc92-4d97-8cac-c2272f91ee40
# ╠═98102564-0b2b-454b-8b2d-1a3c2cde362b
# ╠═a13cd725-56e6-4f6d-a923-ee858856e3e4
# ╟─8a1ee585-5a1e-42b1-b1f2-fbd7a959b564
# ╠═0c3b9ab8-5aa7-4fce-acb2-3761f1dc8650
# ╟─4148fac3-f834-432c-833e-7c5005cb52d5
# ╠═79cfa429-7668-45cb-8691-ccc59238c09a
# ╟─22932754-adb1-4a11-810a-59e72980d16a
# ╠═4a462f59-57d1-4b63-a281-1a82bf250e54
# ╟─ac644379-6ffa-45d1-a95b-0b3ae522e6a7
# ╠═8ac77806-9ae7-43f9-b644-edb4a2d9d38d
