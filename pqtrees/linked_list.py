"""


"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Iterable


@dataclass
class DoublyLinkedNode:
    data: object
    next: Optional['DoublyLinkedNode'] = None
    prev: Optional['DoublyLinkedNode'] = None


T = TypeVar('T')


class LinkedList(Generic[T]):
    head: Optional[DoublyLinkedNode]
    tail: Optional[DoublyLinkedNode]
    size: int


    def __init__(self) -> None:
        super().__init__()
        self.head = self.tail = None
        self.size = 0


    def __iter__(self) -> Iterable[T]:
        for node in self._iter_nodes():
            yield node.data


    def __contains__(self, data: T) -> bool:
        return self._find_node(data) is not None


    def __str__(self) -> str:
        nodes = ', '.join(map(str, self))
        return f'[LL: {len(self)}] ' + nodes


    def __bool__(self) -> bool:
        return len(self) > 0


    def __len__(self) -> int:
        return self.size


    def insert_first(self, data: T) -> DoublyLinkedNode:
        new_node = DoublyLinkedNode(data)

        if not self.head:
            self._insert_first_node(new_node)

        else:
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node

        self.size += 1
        return new_node


    def append(self, data: T) -> DoublyLinkedNode:
        new_node = DoublyLinkedNode(data)

        if not self.head:
            self._insert_first_node(new_node)

        else:
            new_node.prev = self.tail

            self.tail.next = new_node
            self.tail = new_node

        self.size += 1
        return new_node


    def remove(self, data: T) -> Optional[DoublyLinkedNode]:
        node = self._find_node(data)

        if not node or not self:
            return None

        return self._rm_node(node)


    def pop_left(self) -> Optional[T]:
        return self._rm_node(self.head).data


    def pop_right(self) -> Optional[T]:
        return self._rm_node(self.tail).data


    def _insert_first_node(self, new_node: DoublyLinkedNode) -> None:
        self.head = new_node
        self.tail = new_node


    def _iter_nodes(self) -> Iterable[DoublyLinkedNode]:
        cur = self.head
        while cur:
            yield cur
            cur = cur.next


    def _find_node(self, data: T) -> Optional[DoublyLinkedNode]:
        for node in self._iter_nodes():
            if data == node.data:
                return node

        return None


    def _del_only_node(self) -> DoublyLinkedNode:
        only = self.tail
        self.tail = self.head = None
        self.size = 0
        return only


    def _remove_out_of_two(self, node_to_rm: DoublyLinkedNode) -> DoublyLinkedNode:
        node_left = self.tail if node_to_rm == self.head else self.head
        self.size = 1
        self.head = self.tail = node_left
        node_left.prev = node_left.next = None

        return node_to_rm


    def _edge_case_rm(self, node: DoublyLinkedNode) -> DoublyLinkedNode:
        if len(self) == 1:
            return self._del_only_node()
        return self._remove_out_of_two(node)


    def _rm_node(self, node: DoublyLinkedNode) -> DoublyLinkedNode:
        if not self:
            return DoublyLinkedNode(None)

        if len(self) <= 2:
            return self._edge_case_rm(node)

        self.size -= 1

        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        return node


    def copy(self) -> 'LinkedList':
        new_ll = LinkedList()
        for data in self:
            new_ll.append(data)

        return new_ll


if __name__ == '__main__':
    ll = LinkedList()

    ll.insert_first(3)
    ll.insert_first(2)
    ll.insert_first(1)

    ll.append(4)
    ll.append(5)
    ll.append(6)

    ll2 = ll.copy()

    assert list(ll) == list(range(1, 7))
    assert list(ll2) == list(range(1, 7))

    assert 2 in ll and 5 in ll
    assert 7 not in ll and 0 not in ll

    ll.remove(2)
    ll.remove(4)
    ll.remove(6)

    assert len(ll) == 3
    assert list(ll) == list(range(1, 7, 2))

    assert ll.pop_left() == 1
    assert len(ll) == 2
    assert ll.pop_left() == 3
    assert len(ll) == 1
    assert ll.pop_left() == 5
    assert len(ll) == 0
    assert not ll

    assert len(ll2) == 6
    assert ll2.pop_right() == 6
    assert len(ll2) == 5
    assert ll2.pop_left() == 1
    assert len(ll2) == 4
    assert ll2.pop_right() == 5
    assert len(ll2) == 3
    assert ll2.pop_left() == 2
    assert len(ll2) == 2
    assert ll2.pop_right() == 4
    assert len(ll2) == 1
    assert ll2.pop_left() == 3
    assert not ll2
    assert len(ll2) == 0
