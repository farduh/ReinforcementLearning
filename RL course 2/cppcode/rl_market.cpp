#include <iostream>
#include <random>
using namespace std;
class ActionSpace
{
    public:
        int n;
        ActionSpace()
        {   srand(time(NULL));
            n = 2;
        }

        float sample()
        {
            return rand()%2;
        }

};
int main()
{
    // Constructor called
    ActionSpace as;

    // Access values assigned by constructor
    cout << "random = " << as.sample()<<as.sample()<<as.sample()<<as.sample()<<as.sample()<<as.sample()<<as.sample()<<as.sample()<<as.sample()<<as.sample();

    return 0;
}
