"""ATNLPpt_sparseTensors.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ATNLPpt sparse Tensors

"""

import torch as pt

from ANNpt_globalDefs import *

def createEmptySparseTensor(shape):
	sparseZeroTensor = pt.sparse_coo_tensor(indices=pt.empty((len(shape), 0), dtype=pt.long), values=pt.empty(0), size=shape, device=deviceSparse)
	return sparseZeroTensor

def subtractValueFromSparseTensorValues(sparseTensor, value):
	sparseTensor = addValueToSparseTensorValues(sparseTensor, -value)
	sparseTensor.values().clamp_(min=0)
	return sparseTensor
	
def addValueToSparseTensorValues(sparseTensor, value):
	sparseTensor = sparseTensor.coalesce()
	sparseTensor = pt.sparse_coo_tensor(sparseTensor.indices(), sparseTensor.values() + value, sparseTensor.size(), device=deviceSparse)
	sparseTensor = sparseTensor.coalesce()
	return sparseTensor


#replace or multiply element(s) at index with/by new_value
#preconditions: if replace (ie !multiply), assume that the sparse array contains non-zero values at indices_to_update
def modifySparseTensor(sparseTensor, indicesToUpdate, newValue, multiply=False):
	indicesToUpdate = indicesToUpdate.to(sparseTensor.device)
	
	sparseTensor = sparseTensor.coalesce()
	
	# Transpose indicesToUpdate to match dimensions
	indicesToUpdate = indicesToUpdate.t()  # Shape: (batch_size, N)
	
	# Get sparse tensor indices
	sparseIndices = sparseTensor.indices()   # Shape: (batch_size, nnz)
	
	# Expand dimensions to enable broadcasting
	sparseIndicesExpanded = sparseIndices.unsqueeze(2)	   # Shape: (batch_size, nnz, 1)
	indicesToUpdateExpanded = indicesToUpdate.unsqueeze(1) # Shape: (batch_size, 1, N)
	
	# Compare indices
	matches = (sparseIndicesExpanded == indicesToUpdateExpanded).all(dim=0)  # Shape: (nnz, N)
	
	# Identify matches
	matchMask = matches.any(dim=1)  # Shape: (nnz,)
	
	# Update the values at the matched indices
	if(multiply):
		sparseTensor.values()[matchMask] *= newValue
	else:	#replace
		sparseTensor.values()[matchMask] = newValue
		
	return sparseTensor

	
def mergeTensorSlicesSum(originalSparseTensor, sparseSlices, d):
	# Extract indices and values from the original tensor
	originalIndices = originalSparseTensor._indices()
	originalValues = originalSparseTensor._values()

	# Prepare lists for new indices and values
	allIndices = [originalIndices]
	allValues = [originalValues]

	# Process each slice and adjust for the d dimension
	for index, tensorSlice in sparseSlices.items():
		# Create the index tensor for dimension 'd'
		numNonzero = tensorSlice._indices().size(1)
		dIndices = pt.full((1, numNonzero), index, dtype=tensorSlice._indices().dtype, device=deviceSparse)

		# Build the new indices by inserting dIndices at position 'd'
		sliceIndices = tensorSlice._indices()
		before = sliceIndices[:d, :]
		after = sliceIndices[d:, :]
		newIndices = pt.cat([before, dIndices, after], dim=0)

		# Collect the adjusted indices and values
		allIndices.append(newIndices)
		allValues.append(tensorSlice._values())

	# Concatenate all indices and values, including the original tensor's
	finalIndices = pt.cat(allIndices, dim=1)
	finalValues = pt.cat(allValues)

	# Define the final size of the merged tensor, matching the original
	finalSize = originalSparseTensor.size()

	# Create the updated sparse tensor and coalesce to handle duplicates
	mergedSparseTensor = pt.sparse_coo_tensor(finalIndices, finalValues, size=finalSize, device=deviceSparse)

	mergedSparseTensor = mergedSparseTensor.coalesce()
	mergedSparseTensor.values().clamp_(min=0)

	return mergedSparseTensor


def sliceSparseTensorMulti(sparseTensor, sliceDim, sliceIndices):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at given indices,
	without reducing the number of dimensions.

	Args:
		sparseTensor (pt.sparse.FloatTensor): The input sparse tensor.
		sliceDim (int): The dimension along which to slice.
		sliceIndices (pt.Tensor): A 1D tensor of indices to slice.

	Returns:
		pt.sparse.FloatTensor: The sliced sparse tensor with the same number of dimensions.
	"""
	import torch

	sparseTensor = sparseTensor.coalesce()
	
	# Ensure sliceIndices is a 1D tensor and sorted
	sliceIndices = sliceIndices.view(-1).long()
	sliceIndicesSorted, _ = pt.sort(sliceIndices)

	# Get the indices and values from the sparse tensor
	indices = sparseTensor.indices()  # Shape: (ndim, nnz)
	values = sparseTensor.values()	# Shape: (nnz, ...)

	# Get indices along the slicing dimension
	indicesAlongDim = indices[sliceDim]  # Shape: (nnz,)

	# Use searchsorted to find positions in sliceIndices
	positions = pt.searchsorted(sliceIndicesSorted, indicesAlongDim)

	# Check if indicesAlongDim are actually in sliceIndices
	inBounds = positions < len(sliceIndicesSorted)
	matched = inBounds & (sliceIndicesSorted[positions.clamp(max=len(sliceIndicesSorted)-1)] == indicesAlongDim)

	# Mask to select relevant indices and values
	mask = matched

	# Select the indices and values where mask is True
	selectedIndices = indices[:, mask]
	selectedValues = values[mask]

	# Adjust indices along sliceDim
	newIndicesAlongDim = positions[mask]

	# Update the indices along sliceDim
	selectedIndices[sliceDim] = newIndicesAlongDim

	# Adjust the size of the tensor
	newSize = list(sparseTensor.size())
	newSize[sliceDim] = len(sliceIndices)

	# Create the new sparse tensor
	newSparseTensor = pt.sparse_coo_tensor(selectedIndices, selectedValues, size=newSize, device=deviceSparse)

	return newSparseTensor
	
	

def sliceSparseTensor(sparseTensor, sliceDim, sliceIndex):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at a given index.

	Args:
		sparseTensor (pt.sparse.FloatTensor): The input sparse tensor.
		sliceDim (int): The dimension along which to slice.
		sliceIndex (int): The index at which to slice.

	Returns:
		pt.sparse.FloatTensor: The sliced sparse tensor.
	"""
	sparseTensor = sparseTensor.coalesce()	
	
	# Step 1: Extract indices and values
	indices = sparseTensor._indices()  # Shape: (ndim, nnz)
	values = sparseTensor._values()	# Shape: (nnz, ...)

	# Step 2: Create a mask for entries where indices match sliceIndex at sliceDim
	mask = (indices[sliceDim, :] == sliceIndex)

	# Step 3: Filter indices and values using the mask
	filteredIndices = indices[:, mask]
	filteredValues = values[mask]

	# Step 4: Remove the sliceDim from indices
	newIndices = pt.cat((filteredIndices[:sliceDim, :], filteredIndices[sliceDim+1:, :]), dim=0)

	# Step 5: Adjust the size of the new sparse tensor
	originalSize = sparseTensor.size()
	newSize = originalSize[:sliceDim] + originalSize[sliceDim+1:]

	# Step 6: Create the new sparse tensor
	newSparseTensor = pt.sparse_coo_tensor(newIndices, filteredValues, size=newSize, device=deviceSparse)
	newSparseTensor = newSparseTensor.coalesce()  # Ensure the tensor is in canonical form

	return newSparseTensor

def addSparseTensorToFirstDimIndex(A, B, index):

	# Assume A is a sparse 4D tensor and B is a sparse 3D tensor
	AIndices = A._indices()
	AValues = A._values()

	# Step 1: Create a mask for entries where the first dimension index is index
	mask = (AIndices[0] == index)

	# Step 2: Extract indices and values for A[index]
	AIndexIndices = AIndices[1:, mask]
	AIndexValues = AValues[mask]

	# Step 3: Create sparse tensor A[index]
	AIndex = pt.sparse_coo_tensor(AIndexIndices, AIndexValues, size=B.shape, device=deviceSparse)

	# Step 4: Perform the addition A[index] + B
	C = AIndex + B

	# Step 5: Adjust indices to include the first dimension index index
	CIndices = pt.cat([pt.full((1, C._indices().shape[1]), index, dtype=pt.long, device=C.device), C._indices()], dim=0)
	CValues = C._values()

	# Step 6: Remove old entries and add new entries to A
	ARemainingIndices = AIndices[:, ~mask]
	ARemainingValues = AValues[~mask]
	ANewIndices = pt.cat([ARemainingIndices, CIndices], dim=1)
	ANewValues = pt.cat([ARemainingValues, CValues], dim=0)

	# Step 7: Update A
	A = pt.sparse_coo_tensor(ANewIndices, ANewValues, size=A.size(), device=deviceSparse)
	
	return A

def replaceAllSparseTensorElementsAtFirstDimIndex(A, B, index):

	# Get indices and values of A
	AIndices = A._indices()
	AValues = A._values()

	# Create a mask to filter out entries where the first index is index
	mask = AIndices[0] != index

	# Keep only the entries where the first index is not index
	AIndicesFiltered = AIndices[:, mask]
	AValuesFiltered = AValues[mask]

	# Get indices and values of B
	BIndices = B._indices()
	BValues = B._values()

	# Adjust B's indices to align with A's dimensions by prepending a row of index's
	BIndicesAdjusted = pt.cat((pt.full((1, BIndices.size(1)), index, dtype=pt.long, device=B.device), BIndices), dim=0)

	# Concatenate the filtered A indices/values with the adjusted B indices/values
	newIndices = pt.cat((AIndicesFiltered, BIndicesAdjusted), dim=1)
	newValues = pt.cat((AValuesFiltered, BValues), dim=0)

	# Create a new sparse tensor with the updated indices and values
	ANew = pt.sparse_coo_tensor(newIndices, newValues, size=A.size(), device=deviceSparse)
	
	return ANew

def sparseAssign(A, B, *indices):
	"""
	Assigns sparse tensor B into sparse tensor A at positions specified by indices.

	Args:
		A (pt.sparse.Tensor): The target sparse tensor.
		B (pt.sparse.Tensor): The sparse tensor to assign into A.
		*indices: An arbitrary number of indices (int, slice, or 1D tensor).

	Returns:
		pt.sparse.Tensor: The updated sparse tensor.
	"""
	# Ensure A and B are sparse tensors
	if not A.is_sparse or not B.is_sparse:
		raise ValueError("Both A and B must be sparse tensors.")

	# Convert indices to a list for easier manipulation
	indicesList = list(indices)

	# Validate the number of indices
	if len(indicesList) > A.ndim:
		raise IndexError("Too many indices for tensor of dimension {}".format(A.ndim))

	# Pad indices with empty slices if fewer indices are provided
	if len(indicesList) < A.ndim:
		indicesList.extend([slice(None)] * (A.ndim - len(indicesList)))

	# Process indices to compute the new positions for B's indices in A
	newIndices = []
	for dim, idx in enumerate(indicesList):
		if isinstance(idx, int):
			newIdx = B.indices()[dim] + idx
		elif isinstance(idx, slice):
			start = idx.start or 0
			newIdx = B.indices()[dim] + start
		elif pt.is_tensor(idx):
			idx = idx.to(B.indices().device)
			newIdx = idx[B.indices()[dim]]
		else:
			raise TypeError("Invalid index type: {}".format(type(idx)))
		newIndices.append(newIdx)

	# Stack the new indices
	newIndices = pt.stack(newIndices)

	# Concatenate A's and B's indices and values
	combinedIndices = pt.cat([A.indices(), newIndices], dim=1)
	combinedValues = pt.cat([A.values(), B.values()], dim=0)

	# Create a new sparse tensor with the combined indices and values
	newA = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=A.shape, dtype=A.dtype, device=deviceSparse)

	# Coalesce the tensor to sum duplicate indices
	newA = newA.coalesce()

	return newA

def expandSparseTensor(tensor, q, y, newDimSize=None):
	"""
	Inserts a new dimension at position q in the sparse tensor's indices,
	setting the indices in that dimension to y.

	Args:
		tensor (pt.Tensor): Input x-dimensional sparse tensor.
		q (int): Position to insert the new dimension (0 \u2264 q \u2265 x).
		y (int): Index in the new dimension to set.
		newDimSize (int, optional): Size of the new dimension. If None,
			it defaults to y + 1.

	Returns:
		pt.Tensor: New sparse tensor with x+1 dimensions.
	"""
	if not tensor.is_sparse:
		raise ValueError("Input tensor must be a sparse tensor.")

	indices = tensor.indices()  # Shape: (x, nnz)
	values = tensor.values()	# Shape: (nnz,)

	x, nnz = indices.shape

	# Create a row with the new indices set to y
	yRow = pt.full((1, nnz), y, dtype=indices.dtype, device=indices.device)

	# Insert the new dimension at position q
	newIndices = pt.cat((indices[:q], yRow, indices[q:]), dim=0)

	# Determine the new size
	originalSize = list(tensor.size())

	if newDimSize is None:
		if y >= 0:
			newDimSize = y + 1
		else:
			raise ValueError("Negative index y requires specifying newDimSize.")

	if y >= newDimSize or y < -newDimSize:
		raise ValueError(f"Index y={y} is out of bounds for dimension of size {newDimSize}")

	newSize = originalSize[:q] + [newDimSize] + originalSize[q:]

	# Create the new sparse tensor
	newTensor = pt.sparse_coo_tensor(newIndices, values, size=newSize, device=deviceSparse)

	return newTensor

def expandSparseTensorMulti(tensor, q, y, newDimSize=None):
	"""
	Inserts a new dimension at position q in the sparse tensor's indices,
	setting the indices in that dimension to y (list or pt.Tensor of indices).

	Args:
		tensor (pt.Tensor): Input x-dimensional sparse tensor.
		q (int): Position to insert the new dimension (0 \u2264 q \u2265 x).
		y (list or pt.Tensor): Indices in the new dimension to set, length nnz.
		newDimSize (int, optional): Size of the new dimension. If None,
			it defaults to max(y) + 1.

	Returns:
		pt.Tensor: New sparse tensor with x+1 dimensions.
	"""
	if not tensor.is_sparse:
		raise ValueError("Input tensor must be a sparse tensor.")

	indices = tensor.indices()  # Shape: (x, nnz)
	values = tensor.values()	# Shape: (nnz,)

	x, nnz = indices.shape

	# Convert y to a tensor if it's a list
	y = pt.tensor(y, dtype=indices.dtype, device=indices.device).view(1, nnz)

	if y.shape[1] != nnz:
		raise ValueError(f"Length of y ({y.shape[1]}) must be equal to number of non-zero elements (nnz={nnz}).")

	# Insert the new dimension at position q
	newIndices = pt.cat((indices[:q], y, indices[q:]), dim=0)

	# Determine the new size
	originalSize = list(tensor.size())

	if newDimSize is None:
		newDimSize = int(y.max().item()) + 1

	if y.min().item() < -newDimSize or y.max().item() >= newDimSize:
		raise ValueError(f"Indices in y are out of bounds for dimension of size {newDimSize}.")

	newSize = originalSize[:q] + [newDimSize] + originalSize[q:]

	# Create the new sparse tensor
	newTensor = pt.sparse_coo_tensor(newIndices, values, size=newSize, device=deviceSparse)

	return newTensor

def addElementValueToSparseTensor(spTensor, dimensions, v):

	# Example setup: create a sparse len(dimensions) dimensional tensor
	# spTensor is a sparse_coo_tensor with shape [dimensions]
	# Make sure spTensor is in COO format (coalesce it if necessary)
	spTensor = spTensor.coalesce()

	indices = spTensor._indices()  # Shape: [len(dimensions), nnz]
	values = spTensor._values()	# Shape: [nnz]

	targetIndex = pt.tensor(dimensions, dtype=pt.long, device=spTensor.device).unsqueeze(1)  # shape [len(dimensions), 1]

	# Check if this index already exists in the sparse tensor
	mask = (indices == targetIndex).all(dim=0)  # Boolean mask indicating where the match occurs

	if mask.any():
		# The element already exists.
		idx = mask.nonzero().item()  # Extract that single index
		values[idx] += v
	else:
		# The element does not exist; we need to add it.
		newIndices = pt.cat([indices, targetIndex], dim=1)
		newValues = pt.cat([values, pt.tensor([v], dtype=values.dtype, device=values.device)])
		spTensor = pt.sparse_coo_tensor(newIndices, newValues, spTensor.shape, device=spTensor.device).coalesce()

	return spTensor

def sparse_rowwise_max(x):
	"""
	Computes max over dim=1 (columns) for a 2D sparse COO tensor 'x'.
	Returns a 1D tensor of size [x.size(0)] with the max values.
	"""

	# Make sure the tensor is in COO format and has unique indices
	x = x.coalesce()

	# Extract COO indices and values
	indices = x._indices()  # shape: [2, nnz]
	values  = x._values()   # shape: [nnz]

	# Row indices (which row each non-zero belongs to)
	row_idx = indices[0]

	# Initialize output with -inf so we can do a max-reduction
	out = pt.full((x.size(0),), float('-inf'), dtype=values.dtype, device=values.device)

	# Use index_reduce_ (available in newer PyTorch versions) to compute max
	out.index_reduce_(dim=0, index=row_idx, source=values, reduce="amax")

	return out

def selectAindicesContainedInB(A, B):
	# Suppose A and B are sparse tensors of the same shape.
	# Make sure they are coalesced so .indices() and .values() behave nicely.
	A = A.coalesce()
	B = B.coalesce()

	# Extract indices (shape: [ndim, nnz]) and values (shape: [nnz])
	A_indices = A.indices()  # [dim, nnzA]
	A_values  = A.values()   # [nnzA]
	B_indices = B.indices()  # [dim, nnzB]

	# Transpose the indices to shape [nnz, dim] for comparison
	A_indices_t = A_indices.t()  # [nnzA, dim]
	B_indices_t = B_indices.t()  # [nnzB, dim]

	# Compare every index in A to every index in B using broadcasting:
	#  1)  Expand A_indices_t to [nnzA, 1, dim]
	#  2)  Expand B_indices_t to [1, nnzB, dim]
	#  3)  Compare elementwise (==), giving [nnzA, nnzB, dim]
	#  4)  Check all coordinates match with .all(dim=2), giving [nnzA, nnzB]
	#  5)  Reduce along nnzB dimension with .any(dim=1), yielding [nnzA]
	mask = (A_indices_t.unsqueeze(1) == B_indices_t.unsqueeze(0)).all(dim=2).any(dim=1)

	# mask[i] = True if A_indices_t[i] is in B, else False

	# Use the mask to pick out the "intersection" of A's indices
	A_indices_in_B = A_indices[:, mask]
	A_values_in_B  = A_values[mask]

	# Build the new sparse tensor that contains only A's entries whose
	# indices appear in B
	A_intersect_B = pt.sparse_coo_tensor(A_indices_in_B,  A_values_in_B, size=A.shape, device=A.device)

	# Now A_intersect_B has only those (index, value) pairs from A 
	# whose indices are also present in B.
	
	return A_intersect_B


			

