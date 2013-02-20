
#include <stdio.h>
#include <stdlib.h>

#ifndef ALIGNED_ARRAY
#define ALIGNED_ARRAY


template <typename E>
class aligned_array {
private:
	unsigned int capacity;
	unsigned int alignment;
	unsigned int num_items;
	E* items;

public:

	aligned_array ()
	{
		capacity = 0;
		num_items = 0;
		alignment = 1;
		items = NULL;
	}

	aligned_array (const unsigned int alignment)
	{
		capacity = 0;
		num_items = 0;
		this->alignment = alignment;
		items = NULL;
	}

	void push_back (const E& item) 
	{
		if (num_items >= capacity) {
			if (capacity == 0) {
				E* new_items = (E*)_aligned_malloc(sizeof(E)*16, alignment);
				if (new_items == NULL) {
					printf("Error _aligned_malloc failed");
					exit(1);
				}
				items = new_items;
				capacity = 16;
			}
			else {
				E* new_items = (E*)_aligned_malloc(sizeof(E)*capacity*2, alignment);
				if (new_items == NULL) {
					printf("error malloc failed");
					exit(1);
				}
				memcpy(new_items, items, sizeof(E)*capacity);

				_aligned_free(items);

				items = new_items;

				capacity *= 2;
			}
		}

		items[num_items++] = item;
	}

	E& pop (void) {
		return items[--num_items];
	}

	E& operator [] (const unsigned int a) {
		return items[a];
	}

	unsigned int size(void) {
		return num_items;
	}

	void clear (void)
	{
		capacity = 0;
		num_items = 0;
		if (items) _aligned_free(items);
		items = NULL;
	}
};


/*
template <class E>
typedef struct aligned_array {
	E* items;
	unsigned int size;
	unsigned int capacity;
	unsigned int alignment;
} aligned_array<class E>;

template <class E>
aligned_array make_aligned_array (unsigned int alignment)
{
	aligned_array hey = {NULL, 0, 0, alignment};

	return hey;
}

template <class E>
void aligned_array_push_back (aligned_array<E>* ary, E& item) {

	if (ary->size >= ary->capacity) {
		if (ary->capacity = 0) {
			E* new_items = (E*)_aligned_malloc(sizeof(E)*16, ary->alignment);
			if (new_items == NULL) {
				printf("Error _aligned_malloc failed");
				exit(1);
			}
			ary->items = new_items;
			ary->capacity = 16;
		}
		else {
			E* new_items = (E*)_aligned_malloc(sizeof(E)*ary->capacity*2, ary->alignment);
			if (new_items == NULL) {
				printf("error malloc failed");
				exit(1);
			}
			memcpy(new_items, ary->items, sizeof(E)*ary->capacity);

			_aligned_free(ary->items);

			ary->items = new_items;

			ary->capacity *= 2;
		}
	}

	ary->items[ary->size] = item;

	++ary->size;
}

template <class E>
void aligned_array_pop (aligned_array<E>* ary) {
	--ary->size;
}

template <class E>
void delete_aligned_array (aligned_array<E>* ary)
{
	ary->capacity = 0;
	ary->size = 0;
	if (ary->items) _aligned_free(ary->items);
	ary->items = NULL;
}
*/

#endif