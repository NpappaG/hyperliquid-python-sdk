import os
import requests
from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3
from web3.middleware import construct_sign_and_send_raw_middleware
from hyperliquid.utils import constants
from hyperliquid.utils.signing import get_timestamp_ms, sign_l1_action, address_to_bytes

# Connect to the JSON-RPC endpoint
rpc_url = "https://api.hyperliquid-testnet.xyz/evm"
w3 = Web3(Web3.HTTPProvider(rpc_url))

# The account will be used both for deploying the ERC20 contract and linking it to your native spot asset
account: LocalAccount = Account.from_key(os.getenv("0xPK"))  # Change this to your private key
print(f"Running with address {account.address}")
w3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))
w3.eth.default_account = account.address

# Verify connection
if not w3.is_connected():
    raise Exception("Failed to connect to the Ethereum network")

# Customizable parameters
token_name = "Billy"
token_symbol = "BILLY"
initial_supply = 1000000 * 10**18  # 1,000,000 tokens with 18 decimals
decimals = 18

# Define the ABI and Bytecode for the ERC20 contract
purr_abi = {
    "_format": "hh-sol-artifact-1",
    "contractName": "Purr",
    "sourceName": "contracts/Purr.sol",
    "abi": [
        {
            "inputs": [
                {"internalType": "string", "name": "name_", "type": "string"},
                {"internalType": "string", "name": "symbol_", "type": "string"},
                {"internalType": "uint256", "name": "initialSupply", "type": "uint256"},
                {"internalType": "uint8", "name": "decimals_", "type": "uint8"}
            ],
            "stateMutability": "nonpayable",
            "type": "constructor"
        },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        }
      ],
      "name": "Approval",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "previousOwner",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "newOwner",
          "type": "address"
        }
      ],
      "name": "OwnershipTransferred",
      "type": "event"
    },
    {
      "anonymous": False,
      "inputs": [
        {
          "indexed": True,
          "internalType": "address",
          "name": "from",
          "type": "address"
        },
        {
          "indexed": True,
          "internalType": "address",
          "name": "to",
          "type": "address"
        },
        {
          "indexed": False,
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        }
      ],
      "name": "Transfer",
      "type": "event"
    },
    {
      "inputs": [],
      "name": "DOMAIN_SEPARATOR",
      "outputs": [
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        }
      ],
      "name": "allowance",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "amount",
          "type": "uint256"
        }
      ],
      "name": "approve",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "account",
          "type": "address"
        }
      ],
      "name": "balanceOf",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "decimals",
      "outputs": [
        {
          "internalType": "uint8",
          "name": "",
          "type": "uint8"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "subtractedValue",
          "type": "uint256"
        }
      ],
      "name": "decreaseAllowance",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "addedValue",
          "type": "uint256"
        }
      ],
      "name": "increaseAllowance",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "amount",
          "type": "uint256"
        }
      ],
      "name": "mint",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "name",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        }
      ],
      "name": "nonces",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "owner",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "owner",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "spender",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "deadline",
          "type": "uint256"
        },
        {
          "internalType": "uint8",
          "name": "v",
          "type": "uint8"
        },
        {
          "internalType": "bytes32",
          "name": "r",
          "type": "bytes32"
        },
        {
          "internalType": "bytes32",
          "name": "s",
          "type": "bytes32"
        }
      ],
      "name": "permit",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "renounceOwnership",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "symbol",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "totalSupply",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "to",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "amount",
          "type": "uint256"
        }
      ],
      "name": "transfer",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "from",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "to",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "amount",
          "type": "uint256"
        }
      ],
      "name": "transferFrom",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "newOwner",
          "type": "address"
        }
      ],
      "name": "transferOwnership",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
  ],
  "bytecode": "0x6101406040523480156200001257600080fd5b5060405180604001604052806004815260200163282aa92960e11b81525080604051806040016040528060018152602001603160f81b81525060405180604001604052806004815260200163282aa92960e11b81525060405180604001604052806004815260200163282aa92960e11b81525081600390805190602001906200009d929190620001aa565b508051620000b3906004906020840190620001aa565b5050825160209384012082519284019290922060e08390526101008190524660a0818152604080517f8b73c3c69bb8fe3d512ecc4cf759cc79239f7b179b0ffacaa9a75d522b39400f818901819052818301979097526060810194909452608080850193909352308483018190528151808603909301835260c09485019091528151919096012090529290925261012052506200015290503362000158565b6200028d565b600780546001600160a01b038381166001600160a01b0319831681179093556040519116919082907f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e090600090a35050565b828054620001b89062000250565b90600052602060002090601f016020900481019282620001dc576000855562000227565b82601f10620001f757805160ff191683800117855562000227565b8280016001018555821562000227579182015b82811115620002275782518255916020019190600101906200020a565b506200023592915062000239565b5090565b5b808211156200023557600081556001016200023a565b600181811c908216806200026557607f821691505b602082108114156200028757634e487b7160e01b600052602260045260246000fd5b50919050565b60805160a05160c05160e051610100516101205161198c620002dd6000396000610dc701526000610e1601526000610df101526000610d4a01526000610d7401526000610d9e015261198c6000f3fe608060405234801561001057600080fd5b50600436106101365760003560e01c80637ecebe00116100b2578063a457c2d711610081578063d505accf11610066578063d505accf14610287578063dd62ed3e1461029a578063f2fde38b146102e057600080fd5b8063a457c2d714610261578063a9059cbb1461027457600080fd5b80637ecebe001461020b5780638da5cb5b1461021e57806395d89b4114610246578063a0712d681461024e57600080fd5b8063313ce5671161010957806339509351116100ee57806339509351146101b857806370a08231146101cb578063715018a61461020157600080fd5b8063313ce567146101a15780633644e515146101b057600080fd5b806306fdde031461013b578063095ea7b31461015957806318160ddd1461017c57806323b872dd1461018e575b600080fd5b6101436102f3565b6040516101509190611698565b60405180910390f35b61016c610167366004611734565b610385565b6040519015158152602001610150565b6002545b604051908152602001610150565b61016c61019c36600461175e565b61039d565b60405160088152602001610150565b6101806103c1565b61016c6101c6366004611734565b6103d0565b6101806101d936600461179a565b73ffffffffffffffffffffffffffffffffffffffff1660009081526020819052604090205490565b61020961041c565b005b61018061021936600461179a565b610430565b60075460405173ffffffffffffffffffffffffffffffffffffffff9091168152602001610150565b61014361045d565b61020961025c3660046117bc565b61046c565b61016c61026f366004611734565b61049b565b61016c610282366004611734565b610571565b6102096102953660046117d5565b61057f565b6101806102a8366004611848565b73ffffffffffffffffffffffffffffffffffffffff918216600090815260016020908152604080832093909416825291909152205490565b6102096102ee36600461179a565b61073e565b6060600380546103029061187b565b80601f016020809104026020016040519081016040528092919081815260200182805461032e9061187b565b801561037b5780601f106103505761010080835404028352916020019161037b565b820191906000526020600020905b81548152906001019060200180831161035e57829003601f168201915b5050505050905090565b6000336103938185856107f2565b5060019392505050565b6000336103ab8582856109a6565b6103b6858585610a7d565b506001949350505050565b60006103cb610d30565b905090565b33600081815260016020908152604080832073ffffffffffffffffffffffffffffffffffffffff8716845290915281205490919061039390829086906104179087906118f8565b6107f2565b610424610e64565b61042e6000610ee5565b565b73ffffffffffffffffffffffffffffffffffffffff81166000908152600560205260408120545b92915050565b6060600480546103029061187b565b610474610e64565b3360008181526020819052604090205461048e9190610f5c565b6104983382611141565b50565b33600081815260016020908152604080832073ffffffffffffffffffffffffffffffffffffffff8716845290915281205490919083811015610564576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602560248201527f45524332303a2064656372656173656420616c6c6f77616e63652062656c6f7760448201527f207a65726f00000000000000000000000000000000000000000000000000000060648201526084015b60405180910390fd5b6103b682868684036107f2565b600033610393818585610a7d565b834211156105e9576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601d60248201527f45524332305065726d69743a206578706972656420646561646c696e65000000604482015260640161055b565b60007f6e71edae12b1b97f4d1f60370fef10105fa2faae0126114a169c64845d6126c98888886106188c611261565b60408051602081019690965273ffffffffffffffffffffffffffffffffffffffff94851690860152929091166060840152608083015260a082015260c0810186905260e001604051602081830303815290604052805190602001209050600061068082611296565b90506000610690828787876112ff565b90508973ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff1614610727576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601e60248201527f45524332305065726d69743a20696e76616c6964207369676e61747572650000604482015260640161055b565b6107328a8a8a6107f2565b50505050505050505050565b610746610e64565b73ffffffffffffffffffffffffffffffffffffffff81166107e9576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602660248201527f4f776e61626c653a206e6577206f776e657220697320746865207a65726f206160448201527f6464726573730000000000000000000000000000000000000000000000000000606482015260840161055b565b61049881610ee5565b73ffffffffffffffffffffffffffffffffffffffff8316610894576040517f08c379a0000000000000000000000000000000000000000000000000000000008152602060048201526024808201527f45524332303a20617070726f76652066726f6d20746865207a65726f2061646460448201527f7265737300000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff8216610937576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45524332303a20617070726f766520746f20746865207a65726f20616464726560448201527f7373000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff83811660008181526001602090815260408083209487168084529482529182902085905590518481527f8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b92591015b60405180910390a3505050565b73ffffffffffffffffffffffffffffffffffffffff8381166000908152600160209081526040808320938616835292905220547fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8114610a775781811015610a6a576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601d60248201527f45524332303a20696e73756666696369656e7420616c6c6f77616e6365000000604482015260640161055b565b610a7784848484036107f2565b50505050565b73ffffffffffffffffffffffffffffffffffffffff8316610b20576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602560248201527f45524332303a207472616e736665722066726f6d20746865207a65726f20616460448201527f6472657373000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff8216610bc3576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602360248201527f45524332303a207472616e7366657220746f20746865207a65726f206164647260448201527f6573730000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff831660009081526020819052604090205481811015610c79576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602660248201527f45524332303a207472616e7366657220616d6f756e742065786365656473206260448201527f616c616e63650000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff808516600090815260208190526040808220858503905591851681529081208054849290610cbd9084906118f8565b925050819055508273ffffffffffffffffffffffffffffffffffffffff168473ffffffffffffffffffffffffffffffffffffffff167fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef84604051610d2391815260200190565b60405180910390a3610a77565b60003073ffffffffffffffffffffffffffffffffffffffff7f000000000000000000000000000000000000000000000000000000000000000016148015610d9657507f000000000000000000000000000000000000000000000000000000000000000046145b15610dc057507f000000000000000000000000000000000000000000000000000000000000000090565b50604080517f00000000000000000000000000000000000000000000000000000000000000006020808301919091527f0000000000000000000000000000000000000000000000000000000000000000828401527f000000000000000000000000000000000000000000000000000000000000000060608301524660808301523060a0808401919091528351808403909101815260c0909201909252805191012090565b60075473ffffffffffffffffffffffffffffffffffffffff16331461042e576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820181905260248201527f4f776e61626c653a2063616c6c6572206973206e6f7420746865206f776e6572604482015260640161055b565b6007805473ffffffffffffffffffffffffffffffffffffffff8381167fffffffffffffffffffffffff0000000000000000000000000000000000000000831681179093556040519116919082907f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e090600090a35050565b73ffffffffffffffffffffffffffffffffffffffff8216610fff576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602160248201527f45524332303a206275726e2066726f6d20746865207a65726f2061646472657360448201527f7300000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff8216600090815260208190526040902054818110156110b5576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45524332303a206275726e20616d6f756e7420657863656564732062616c616e60448201527f6365000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff831660009081526020819052604081208383039055600280548492906110f1908490611910565b909155505060405182815260009073ffffffffffffffffffffffffffffffffffffffff8516907fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef90602001610999565b73ffffffffffffffffffffffffffffffffffffffff82166111be576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601f60248201527f45524332303a206d696e7420746f20746865207a65726f206164647265737300604482015260640161055b565b80600260008282546111d091906118f8565b909155505073ffffffffffffffffffffffffffffffffffffffff82166000908152602081905260408120805483929061120a9084906118f8565b909155505060405181815273ffffffffffffffffffffffffffffffffffffffff8316906000907fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef9060200160405180910390a35050565b73ffffffffffffffffffffffffffffffffffffffff811660009081526005602052604090208054600181018255905b50919050565b60006104576112a3610d30565b836040517f19010000000000000000000000000000000000000000000000000000000000006020820152602281018390526042810182905260009060620160405160208183030381529060405280519060200120905092915050565b600080600061131087878787611327565b9150915061131d8161143f565b5095945050505050565b6000807f7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a083111561135e5750600090506003611436565b8460ff16601b1415801561137657508460ff16601c14155b156113875750600090506004611436565b6040805160008082526020820180845289905260ff881692820192909252606081018690526080810185905260019060a0016020604051602081039080840390855afa1580156113db573d6000803e3d6000fd5b50506040517fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0015191505073ffffffffffffffffffffffffffffffffffffffff811661142f57600060019250925050611436565b9150600090505b94509492505050565b600081600481111561145357611453611927565b141561145c5750565b600181600481111561147057611470611927565b14156114d8576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601860248201527f45434453413a20696e76616c6964207369676e61747572650000000000000000604482015260640161055b565b60028160048111156114ec576114ec611927565b1415611554576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601f60248201527f45434453413a20696e76616c6964207369676e6174757265206c656e67746800604482015260640161055b565b600381600481111561156857611568611927565b14156115f6576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45434453413a20696e76616c6964207369676e6174757265202773272076616c60448201527f7565000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b600481600481111561160a5761160a611927565b1415610498576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45434453413a20696e76616c6964207369676e6174757265202776272076616c60448201527f7565000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b600060208083528351808285015260005b818110156116c5578581018301518582016040015282016116a9565b818111156116d7576000604083870101525b50601f017fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe016929092016040019392505050565b803573ffffffffffffffffffffffffffffffffffffffff8116811461172f57600080fd5b919050565b6000806040838503121561174757600080fd5b6117508361170b565b946020939093013593505050565b60008060006060848603121561177357600080fd5b61177c8461170b565b925061178a6020850161170b565b9150604084013590509250925092565b6000602082840312156117ac57600080fd5b6117b58261170b565b9392505050565b6000602082840312156117ce57600080fd5b5035919050565b600080600080600080600060e0888a0312156117f057600080fd5b6117f98861170b565b96506118076020890161170b565b95506040880135945060608801359350608088013560ff8116811461182b57600080fd5b9699959850939692959460a0840135945060c09093013592915050565b6000806040838503121561185b57600080fd5b6118648361170b565b91506118726020840161170b565b90509250929050565b600181811c9082168061188f57607f821691505b60208210811415611290577f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b6000821982111561190b5761190b6118c9565b500190565b600082821015611922576119226118c9565b500390565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602160045260246000fdfea26469706673582212205730b41eb3bf6127251dbb860cb96d8de333b02d2311eabf14197cc48a2fafd464736f6c63430008090033",
  "deployedBytecode": "0x608060405234801561001057600080fd5b50600436106101365760003560e01c80637ecebe00116100b2578063a457c2d711610081578063d505accf11610066578063d505accf14610287578063dd62ed3e1461029a578063f2fde38b146102e057600080fd5b8063a457c2d714610261578063a9059cbb1461027457600080fd5b80637ecebe001461020b5780638da5cb5b1461021e57806395d89b4114610246578063a0712d681461024e57600080fd5b8063313ce5671161010957806339509351116100ee57806339509351146101b857806370a08231146101cb578063715018a61461020157600080fd5b8063313ce567146101a15780633644e515146101b057600080fd5b806306fdde031461013b578063095ea7b31461015957806318160ddd1461017c57806323b872dd1461018e575b600080fd5b6101436102f3565b6040516101509190611698565b60405180910390f35b61016c610167366004611734565b610385565b6040519015158152602001610150565b6002545b604051908152602001610150565b61016c61019c36600461175e565b61039d565b60405160088152602001610150565b6101806103c1565b61016c6101c6366004611734565b6103d0565b6101806101d936600461179a565b73ffffffffffffffffffffffffffffffffffffffff1660009081526020819052604090205490565b61020961041c565b005b61018061021936600461179a565b610430565b60075460405173ffffffffffffffffffffffffffffffffffffffff9091168152602001610150565b61014361045d565b61020961025c3660046117bc565b61046c565b61016c61026f366004611734565b61049b565b61016c610282366004611734565b610571565b6102096102953660046117d5565b61057f565b6101806102a8366004611848565b73ffffffffffffffffffffffffffffffffffffffff918216600090815260016020908152604080832093909416825291909152205490565b6102096102ee36600461179a565b61073e565b6060600380546103029061187b565b80601f016020809104026020016040519081016040528092919081815260200182805461032e9061187b565b801561037b5780601f106103505761010080835404028352916020019161037b565b820191906000526020600020905b81548152906001019060200180831161035e57829003601f168201915b5050505050905090565b6000336103938185856107f2565b5060019392505050565b6000336103ab8582856109a6565b6103b6858585610a7d565b506001949350505050565b60006103cb610d30565b905090565b33600081815260016020908152604080832073ffffffffffffffffffffffffffffffffffffffff8716845290915281205490919061039390829086906104179087906118f8565b6107f2565b610424610e64565b61042e6000610ee5565b565b73ffffffffffffffffffffffffffffffffffffffff81166000908152600560205260408120545b92915050565b6060600480546103029061187b565b610474610e64565b3360008181526020819052604090205461048e9190610f5c565b6104983382611141565b50565b33600081815260016020908152604080832073ffffffffffffffffffffffffffffffffffffffff8716845290915281205490919083811015610564576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602560248201527f45524332303a2064656372656173656420616c6c6f77616e63652062656c6f7760448201527f207a65726f00000000000000000000000000000000000000000000000000000060648201526084015b60405180910390fd5b6103b682868684036107f2565b600033610393818585610a7d565b834211156105e9576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601d60248201527f45524332305065726d69743a206578706972656420646561646c696e65000000604482015260640161055b565b60007f6e71edae12b1b97f4d1f60370fef10105fa2faae0126114a169c64845d6126c98888886106188c611261565b60408051602081019690965273ffffffffffffffffffffffffffffffffffffffff94851690860152929091166060840152608083015260a082015260c0810186905260e001604051602081830303815290604052805190602001209050600061068082611296565b90506000610690828787876112ff565b90508973ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff1614610727576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601e60248201527f45524332305065726d69743a20696e76616c6964207369676e61747572650000604482015260640161055b565b6107328a8a8a6107f2565b50505050505050505050565b610746610e64565b73ffffffffffffffffffffffffffffffffffffffff81166107e9576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602660248201527f4f776e61626c653a206e6577206f776e657220697320746865207a65726f206160448201527f6464726573730000000000000000000000000000000000000000000000000000606482015260840161055b565b61049881610ee5565b73ffffffffffffffffffffffffffffffffffffffff8316610894576040517f08c379a0000000000000000000000000000000000000000000000000000000008152602060048201526024808201527f45524332303a20617070726f76652066726f6d20746865207a65726f2061646460448201527f7265737300000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff8216610937576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45524332303a20617070726f766520746f20746865207a65726f20616464726560448201527f7373000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff83811660008181526001602090815260408083209487168084529482529182902085905590518481527f8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b92591015b60405180910390a3505050565b73ffffffffffffffffffffffffffffffffffffffff8381166000908152600160209081526040808320938616835292905220547fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8114610a775781811015610a6a576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601d60248201527f45524332303a20696e73756666696369656e7420616c6c6f77616e6365000000604482015260640161055b565b610a7784848484036107f2565b50505050565b73ffffffffffffffffffffffffffffffffffffffff8316610b20576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602560248201527f45524332303a207472616e736665722066726f6d20746865207a65726f20616460448201527f6472657373000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff8216610bc3576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602360248201527f45524332303a207472616e7366657220746f20746865207a65726f206164647260448201527f6573730000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff831660009081526020819052604090205481811015610c79576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602660248201527f45524332303a207472616e7366657220616d6f756e742065786365656473206260448201527f616c616e63650000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff808516600090815260208190526040808220858503905591851681529081208054849290610cbd9084906118f8565b925050819055508273ffffffffffffffffffffffffffffffffffffffff168473ffffffffffffffffffffffffffffffffffffffff167fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef84604051610d2391815260200190565b60405180910390a3610a77565b60003073ffffffffffffffffffffffffffffffffffffffff7f000000000000000000000000000000000000000000000000000000000000000016148015610d9657507f000000000000000000000000000000000000000000000000000000000000000046145b15610dc057507f000000000000000000000000000000000000000000000000000000000000000090565b50604080517f00000000000000000000000000000000000000000000000000000000000000006020808301919091527f0000000000000000000000000000000000000000000000000000000000000000828401527f000000000000000000000000000000000000000000000000000000000000000060608301524660808301523060a0808401919091528351808403909101815260c0909201909252805191012090565b60075473ffffffffffffffffffffffffffffffffffffffff16331461042e576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820181905260248201527f4f776e61626c653a2063616c6c6572206973206e6f7420746865206f776e6572604482015260640161055b565b6007805473ffffffffffffffffffffffffffffffffffffffff8381167fffffffffffffffffffffffff0000000000000000000000000000000000000000831681179093556040519116919082907f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e090600090a35050565b73ffffffffffffffffffffffffffffffffffffffff8216610fff576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602160248201527f45524332303a206275726e2066726f6d20746865207a65726f2061646472657360448201527f7300000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff8216600090815260208190526040902054818110156110b5576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45524332303a206275726e20616d6f756e7420657863656564732062616c616e60448201527f6365000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b73ffffffffffffffffffffffffffffffffffffffff831660009081526020819052604081208383039055600280548492906110f1908490611910565b909155505060405182815260009073ffffffffffffffffffffffffffffffffffffffff8516907fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef90602001610999565b73ffffffffffffffffffffffffffffffffffffffff82166111be576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601f60248201527f45524332303a206d696e7420746f20746865207a65726f206164647265737300604482015260640161055b565b80600260008282546111d091906118f8565b909155505073ffffffffffffffffffffffffffffffffffffffff82166000908152602081905260408120805483929061120a9084906118f8565b909155505060405181815273ffffffffffffffffffffffffffffffffffffffff8316906000907fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef9060200160405180910390a35050565b73ffffffffffffffffffffffffffffffffffffffff811660009081526005602052604090208054600181018255905b50919050565b60006104576112a3610d30565b836040517f19010000000000000000000000000000000000000000000000000000000000006020820152602281018390526042810182905260009060620160405160208183030381529060405280519060200120905092915050565b600080600061131087878787611327565b9150915061131d8161143f565b5095945050505050565b6000807f7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a083111561135e5750600090506003611436565b8460ff16601b1415801561137657508460ff16601c14155b156113875750600090506004611436565b6040805160008082526020820180845289905260ff881692820192909252606081018690526080810185905260019060a0016020604051602081039080840390855afa1580156113db573d6000803e3d6000fd5b50506040517fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0015191505073ffffffffffffffffffffffffffffffffffffffff811661142f57600060019250925050611436565b9150600090505b94509492505050565b600081600481111561145357611453611927565b141561145c5750565b600181600481111561147057611470611927565b14156114d8576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601860248201527f45434453413a20696e76616c6964207369676e61747572650000000000000000604482015260640161055b565b60028160048111156114ec576114ec611927565b1415611554576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152601f60248201527f45434453413a20696e76616c6964207369676e6174757265206c656e67746800604482015260640161055b565b600381600481111561156857611568611927565b14156115f6576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45434453413a20696e76616c6964207369676e6174757265202773272076616c60448201527f7565000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b600481600481111561160a5761160a611927565b1415610498576040517f08c379a000000000000000000000000000000000000000000000000000000000815260206004820152602260248201527f45434453413a20696e76616c6964207369676e6174757265202776272076616c60448201527f7565000000000000000000000000000000000000000000000000000000000000606482015260840161055b565b600060208083528351808285015260005b818110156116c5578581018301518582016040015282016116a9565b818111156116d7576000604083870101525b50601f017fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe016929092016040019392505050565b803573ffffffffffffffffffffffffffffffffffffffff8116811461172f57600080fd5b919050565b6000806040838503121561174757600080fd5b6117508361170b565b946020939093013593505050565b60008060006060848603121561177357600080fd5b61177c8461170b565b925061178a6020850161170b565b9150604084013590509250925092565b6000602082840312156117ac57600080fd5b6117b58261170b565b9392505050565b6000602082840312156117ce57600080fd5b5035919050565b600080600080600080600060e0888a0312156117f057600080fd5b6117f98861170b565b96506118076020890161170b565b95506040880135945060608801359350608088013560ff8116811461182b57600080fd5b9699959850939692959460a0840135945060c09093013592915050565b6000806040838503121561185b57600080fd5b6118648361170b565b91506118726020840161170b565b90509250929050565b600181811c9082168061188f57607f821691505b60208210811415611290577f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b6000821982111561190b5761190b6118c9565b500190565b600082821015611922576119226118c9565b500390565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602160045260246000fdfea26469706673582212205730b41eb3bf6127251dbb860cb96d8de333b02d2311eabf14197cc48a2fafd464736f6c63430008090033",
  "linkReferences": {},
  "deployedLinkReferences": {}
}
# Create the contract instance
Purr = w3.eth.contract(abi=purr_abi['abi'], bytecode=purr_abi['bytecode'])

# Deploy the contract
should_deploy_contract = True
should_link_contract = True

if should_deploy_contract:
    tx_hash = Purr.constructor(token_name, token_symbol, initial_supply, decimals).transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt.contractAddress
    print(f"Contract deployed at address: {contract_address}")

    # Mint initial supply and transfer to system address
    purr = w3.eth.contract(address=contract_address, abi=purr_abi['abi'])
    tx_hash = purr.functions.mint(initial_supply).transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    tx_hash = purr.functions.transfer('0x2222222222222222222222222222222222222222', initial_supply).transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
else:
    contract_address = "0x8cDE56336E289c028C8f7CF5c20283fF02272182"

# Link the token to the EVM
if should_link_contract:
    action = {
        "type": "spotDeploy",
        "setEvmContract": {
            "token": 1,
            "address": contract_address.lower(),
            "evmExtraWeiDecimals": 13,
        },
    }
    nonce = get_timestamp_ms()
    signature = sign_l1_action(account, action, None, nonce, False)
    payload = {
        "action": action,
        "nonce": nonce,
        "signature": signature,
        "vaultAddress": None,
    }
    response = requests.post(constants.TESTNET_API_URL + "/exchange", json=payload)
    print(response.json())