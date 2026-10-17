package flat

// maxHeap keeps the farthest Result on top. Use it through container/heap.
type maxHeap []Result

func (h maxHeap) Len() int {
	return len(h)
}

func (h maxHeap) Less(i, j int) bool {
	return h[i].Dist > h[j].Dist
}

func (h maxHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *maxHeap) Push(x any) {
	*h = append(*h, x.(Result))
}

// Pop removes the last element: heap.Pop has already swapped the top there.
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
