class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    # Add a node to the end of the list
    def append(self, data):
        new_node = Node(data)
        if not self.head:  # If the list is empty
            self.head = new_node
            return
        current = self.head
        while current.next:  # Traverse to the end
            current = current.next
        current.next = new_node

    # Print the entire list
    def print_list(self):
        if not self.head:
            print("The list is empty.")
            return
        current = self.head
        print("Linked List:", end=" ")
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    # Delete the nth node (1-based index)
    def delete_nth_node(self, n):
        try:
            if not self.head:
                raise Exception("Cannot delete from an empty list.")

            if n <= 0:
                raise Exception("Index should be 1 or greater.")

            if n == 1:
                self.head = self.head.next
                return

            current = self.head
            count = 1
            while current and count < n - 1:
                current = current.next
                count += 1

            if not current or not current.next:
                raise Exception("Index out of range.")

            current.next = current.next.next

        except Exception as e:
            print(f"Error: {e}")


# 🔁 Sample Test
if __name__ == "__main__":
    ll = LinkedList()

    # Add some nodes
    ll.append(10)
    ll.append(20)
    ll.append(30)
    ll.append(40)

    # Print initial list
    ll.print_list()  # 10 -> 20 -> 30 -> 40 -> None

    # Delete 3rd node (30)
    ll.delete_nth_node(3)
    ll.print_list()  # 10 -> 20 -> 40 -> None

    # Try deleting an out-of-range node
    ll.delete_nth_node(10)  # Should handle gracefully

    # Try deleting from index 0 (invalid)
    ll.delete_nth_node(0)   # Should handle gracefully

    # Delete all nodes
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)

    # Try deleting from empty list
    ll.delete_nth_node(1)   # Should show error

    # Final print
    ll.print_list()  # Should say list is empty
