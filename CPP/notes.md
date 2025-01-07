## Namespaces

### 2025eko urtarrilaren 7an

A namespace provides a solution for preventing name conflicts in large projects.

Each entity needs a different name unless we use a namespace:

namespace first{
    int x = 0;
}

namespace second{
    int x = 0;
}

then, the next becomes valid

cout << x;

cout << first::x;

cout << second::x;

The "::" operator is known as the scope resolution operator.   

In order to avoid messing the standard namespace with others, one can use lines like *using std::cout;*

### Typedef keyword

Allows you to give aliases to data types:

typedef std::string str;

std::string Villain = "Decima Technologies"; becomes str Villain = "Decima Technologies";.     

This can also be done by:

using str = std::string; 
str Villain = "Decima Technologies";

### Explicit casting

To explicitly cast a value in another type:

(char) 100;