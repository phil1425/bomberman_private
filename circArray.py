"""
    Implementation of a circular buffer of fixed storage size.
    Author: George Heineman
"""
import numpy as np

class CircularArray:
    
    def __init__(self, shape):
        """Store buffer in given storage."""
        self.buffer = np.zeros(shape)
        self.populated = np.zeros(shape[0])
        self.low = 0
        self.high = 0
        self.size = shape[0]
        self.count = 0

    def isEmpty(self):
        """Determines if buffer is empty."""
        return self.count == 0

    def isFull(self):
        """Determines if buffer is full."""
        return self.count == self.size
        
    def __len__(self):
        """Returns number of elements in buffer."""
        return self.count
        
    def add(self, value):
        """Adds value to buffer, overwrite as needed."""
        if self.isFull():
            self.low = (self.low+1) % self.size
        else:
            self.count += 1
        self.buffer[self.high] = value
        self.populated[self.high] = 1
        self.high = (self.high + 1) % self.size
    
    def remove(self):
        """Removes oldest value from non-empty buffer."""
        if self.count == 0:
            raise Exception ("Circular Buffer is empty");
        value = self.buffer[self.low]
        self.populated[self.low] = 0
        self.low = (self.low + 1) % self.size
        self.count -= 1
        return value

    def as_array(self):
        if self.isFull():
            return self.buffer
        else:
            return self.buffer[self.populated==1]
    
    def recall(self):
        out_array = np.zeros(self.count)
        for i, item in enumerate(self):
            out_array[i] = item
        return out_array

    
    def __iter__(self):
        """Return elements in the circular buffer in order using iterator."""
        idx = self.low
        num = self.count
        while num > 0:
            yield self.buffer[idx]
            idx = (idx + 1) % self.size
            num -= 1

    def __repr__(self):
        """String representation of circular buffer."""
        if self.isEmpty():
            return 'cb:[]'

        return 'cb:[' + ','.join(map(str,self)) + ']'